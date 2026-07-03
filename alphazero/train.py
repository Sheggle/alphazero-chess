"""AlphaZero training loop for tic-tac-toe.

Each iteration:
  1. Self-play: generate games with the current net (Dirichlet noise + temp).
  2. Store samples (D4-augmented) in a replay buffer.
  3. Train: SGD/Adam on value-MSE + policy-cross-entropy.
  4. Evaluate: optimal-move rate vs the solver, and a match vs random.

The signal we care about is `optimal_move_rate` -> 1.0 (perfect play everywhere)
and never losing to random.
"""

from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .agents import AZAgent, GumbelAgent, RandomAgent, RawNetAgent
from .arena import play_match
from .encoder import symmetries
from .evaluate import all_nonterminal_states, optimal_move_rate
from .net import NetEvaluator, TicTacToeNet
from .selfplay import play_selfplay_game, play_selfplay_game_gumbel


@dataclass
class Config:
    iterations: int = 25
    games_per_iter: int = 40
    n_sims: int = 75
    c_puct: float = 1.5
    temp_moves: int = 3
    buffer_size: int = 30000
    batch_size: int = 256
    train_steps: int = 150
    lr: float = 1e-3
    weight_decay: float = 1e-4
    channels: int = 32
    device: str = "cpu"
    eval_random_games: int = 200
    eval_mcts_sample: int = 800  # states sampled for MCTS opt-rate (if not full)
    eval_mcts_full: bool = False  # eval opt(mcts) over ALL states each iter
    seed: int = 0
    # --- sims: split self-play vs eval (fall back to n_sims if None) ---
    selfplay_sims: int | None = None
    eval_sims: int | None = None
    # --- Gumbel search (acting + completed-Q target) ---
    use_gumbel: bool = False
    gumbel_max_considered: int = 8
    c_visit: float = 50.0
    c_scale: float = 1.0
    # Mix a fraction of stronger (more sims, wider) self-play games into the data
    # to calibrate the value head on the wider candidate set (cheap robustness).
    mix_frac: float = 0.0
    mix_sims: int = 10
    mix_max_considered: int = 8
    # --- stability / regularization knobs (default = off) ---
    entropy_coef: float = 0.0   # add -coef * H(policy) to loss (anti-collapse)
    value_coef: float = 1.0     # weight on value-MSE term
    log: list = field(default_factory=list)

    @property
    def sp_sims(self) -> int:
        return self.selfplay_sims if self.selfplay_sims is not None else self.n_sims

    @property
    def ev_sims(self) -> int:
        return self.eval_sims if self.eval_sims is not None else self.n_sims


def _encode_samples(samples):
    """Expand self-play samples into augmented (planes, pi, z) training tuples."""
    from .encoder import encode
    out = []
    for s in samples:
        planes = encode(s.state)
        for p, pi in symmetries(planes, s.pi):
            out.append((p, pi, s.z))
    return out


def train(cfg: Config | None = None, verbose: bool = True):
    cfg = cfg or Config()
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)  # the batch sampler uses the global `random` module
    rng = np.random.default_rng(cfg.seed)

    net = TicTacToeNet(channels=cfg.channels)
    evaluator = NetEvaluator(net, device=cfg.device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    buffer: deque = deque(maxlen=cfg.buffer_size)
    eval_states = all_nonterminal_states()
    sample_rng = random.Random(cfg.seed + 7)
    if cfg.eval_mcts_full:
        mcts_eval_states = eval_states
    else:
        mcts_eval_states = sample_rng.sample(eval_states, min(cfg.eval_mcts_sample, len(eval_states)))

    for it in range(1, cfg.iterations + 1):
        t0 = time.time()
        # --- self-play ---
        n_new = 0
        for _ in range(cfg.games_per_iter):
            if cfg.use_gumbel:
                # Most games at the cheap base width; a `mix_frac` minority at a
                # higher sim count / wider width to calibrate the value head.
                if cfg.mix_frac > 0 and rng.random() < cfg.mix_frac:
                    g_sims, g_mc = cfg.mix_sims, cfg.mix_max_considered
                else:
                    g_sims, g_mc = cfg.sp_sims, cfg.gumbel_max_considered
                samples = play_selfplay_game_gumbel(
                    evaluator, n_sims=g_sims, max_considered=g_mc,
                    c_visit=cfg.c_visit, c_scale=cfg.c_scale, c_puct=cfg.c_puct, rng=rng,
                )
            else:
                samples = play_selfplay_game(
                    evaluator, n_sims=cfg.sp_sims, c_puct=cfg.c_puct,
                    temp_moves=cfg.temp_moves, rng=rng,
                )
            enc = _encode_samples(samples)
            buffer.extend(enc)
            n_new += len(enc)

        # --- train ---
        net.train()
        losses = []
        if len(buffer) >= cfg.batch_size:
            for _ in range(cfg.train_steps):
                batch = random.sample(buffer, cfg.batch_size)
                planes = torch.from_numpy(np.stack([b[0] for b in batch])).to(cfg.device)
                target_pi = torch.from_numpy(np.stack([b[1] for b in batch])).to(cfg.device)
                target_z = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=cfg.device)

                logits, value = net(planes)
                logp = F.log_softmax(logits, dim=1)
                policy_loss = -(target_pi * logp).sum(dim=1).mean()
                value_loss = F.mse_loss(value, target_z)
                loss = policy_loss + cfg.value_coef * value_loss
                if cfg.entropy_coef > 0:
                    # Maximize policy entropy => subtract H from loss. Keeps the
                    # policy from collapsing to a near one-hot under overtraining.
                    p = logp.exp()
                    entropy = -(p * logp).sum(dim=1).mean()
                    loss = loss - cfg.entropy_coef * entropy

                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append((policy_loss.item(), value_loss.item()))

        # --- evaluate ---
        if cfg.use_gumbel:
            az_agent = GumbelAgent(evaluator, n_sims=cfg.ev_sims,
                                   max_considered=cfg.gumbel_max_considered,
                                   c_visit=cfg.c_visit, c_scale=cfg.c_scale,
                                   c_puct=cfg.c_puct, rng=np.random.default_rng(cfg.seed + 999))
        else:
            az_agent = AZAgent(evaluator, n_sims=cfg.ev_sims, c_puct=cfg.c_puct,
                               rng=np.random.default_rng(cfg.seed + 999))
        raw_agent = RawNetAgent(evaluator)
        opt_rate_mcts = optimal_move_rate(az_agent, mcts_eval_states)
        opt_rate_raw = optimal_move_rate(raw_agent, eval_states)
        vs_rand = play_match(az_agent, RandomAgent(random.Random(cfg.seed + it)),
                             n_games=cfg.eval_random_games)

        pl = np.mean([l[0] for l in losses]) if losses else float("nan")
        vl = np.mean([l[1] for l in losses]) if losses else float("nan")
        rec = {
            "iter": it, "buffer": len(buffer), "policy_loss": pl, "value_loss": vl,
            "opt_rate_mcts": opt_rate_mcts, "opt_rate_raw": opt_rate_raw,
            "vs_random_score": vs_rand.score, "vs_random_losses": vs_rand.losses,
            "secs": time.time() - t0,
        }
        cfg.log.append(rec)
        if verbose:
            print(
                f"it {it:2d} | buf {len(buffer):5d} | ploss {pl:.3f} vloss {vl:.3f} "
                f"| opt(mcts) {opt_rate_mcts:.3f} opt(raw) {opt_rate_raw:.3f} "
                f"| vs_rand score {vs_rand.score:.3f} losses {vs_rand.losses} "
                f"| {rec['secs']:.1f}s"
            )

    return net, evaluator, cfg


def robust_lowsim_config(iterations: int = 50) -> Config:
    """Recommended config from OPTIMIZATION_LOG.md: strong + overtraining-stable
    at just 3 simulations (Gumbel acting + completed-Q target + entropy reg)."""
    return Config(
        iterations=iterations,
        games_per_iter=40,
        selfplay_sims=3,
        eval_sims=3,
        use_gumbel=True,
        # max_considered must be <= selfplay_sims so the root candidate WIDTH is
        # identical in self-play and eval. The value head is only calibrated on
        # the moves backed up during self-play (the top-`width` policy moves); if
        # eval widens the candidate set it queries the value head out-of-
        # distribution on low-policy moves it over-rates, and search walks into
        # them. With width fixed, eval sims is a pure depth knob (deep eval at
        # width 3 draws a perfect opponent; see OPTIMIZATION_LOG.md Change 4).
        gumbel_max_considered=3,
        c_visit=50.0,
        c_scale=1.0,
        entropy_coef=0.02,
        train_steps=150,
        batch_size=256,
        lr=1e-3,
        weight_decay=1e-4,
        eval_mcts_full=True,
        eval_random_games=100,
        seed=0,
    )


if __name__ == "__main__":
    train()
