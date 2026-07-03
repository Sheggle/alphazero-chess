"""AlphaZero training for 3D Connect 4 (Score Four, 4x4x4).

Reuses the generic Gumbel search (`alphazero.gumbel.GumbelMCTS`) and the exact
same loss as chess training (policy CE toward the Gumbel completed-Q pi, value
MSE, small entropy bonus). The game is tiny, so pure-Python self-play with
batched GPU training is plenty fast.

Run:
    uv run python scripts/train_connect4.py smoke      # quick CPU sanity run
    uv run python scripts/train_connect4.py            # full run (models/connect4)

Eval reports win-rate vs a RANDOM player and vs a 1-ply TACTIC player (takes an
immediate win / blocks an immediate loss), both colors.
"""

from __future__ import annotations

import json
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

# Make `alphazero` importable when run as a bare script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import torch
import torch.nn.functional as F

from alphazero.connect4_encode import encode_state
from alphazero.connect4_env import ACTION_SIZE, Connect4Game
from alphazero.connect4_net import Connect4Evaluator, Connect4Net
from alphazero.gumbel import GumbelMCTS


# ------------------------------------------------------------------ self-play

def play_game(evaluator, sims, mc, c_visit, c_scale, rng, max_ply=64):
    """One self-play game. Returns (samples, stats). Each sample is sparse:
    (planes float16, pi_indices int16, pi_values float32, z float32).

    Connect-4 games terminate quickly (win or full-board draw), so the value
    target is simply the true game outcome from each state's mover perspective.
    """
    g = Connect4Game()
    recs = []
    while not g.is_terminal() and g.ply < max_ply:
        a, pi = GumbelMCTS(evaluator, n_sims=sims, max_considered=mc,
                           c_visit=c_visit, c_scale=c_scale, rng=rng,
                           solve_children=True).run(g, add_noise=True)
        recs.append((encode_state(g), pi, g.to_play))
        g = g.apply(int(a))

    z_p1 = g.result()  # +1/-1/0 in player +1's perspective (0 if draw/capped)
    samples = []
    for planes, pi, to_play in recs:
        z = float(z_p1 * to_play)
        idx = np.nonzero(pi)[0].astype(np.int16)
        samples.append((planes.astype(np.float16), idx, pi[idx].astype(np.float32), np.float32(z)))
    stats = {"plies": g.ply, "result": z_p1, "decisive": z_p1 != 0}
    return samples, stats


# ------------------------------------------ worker plumbing (spawn-safe)

_WORKER = {}


def _init_worker(channels, blocks):
    torch.set_num_threads(1)
    net = Connect4Net(channels=channels, blocks=blocks)
    _WORKER["net"] = net
    _WORKER["ev"] = Connect4Evaluator(net)


def _worker_play(args):
    (weights_path, n_games, sims, mc, c_visit, c_scale, max_ply, seed) = args
    sd = torch.load(weights_path, map_location="cpu", weights_only=True)
    _WORKER["net"].load_state_dict(sd)
    rng = np.random.default_rng(seed)
    out, stats = [], []
    for _ in range(n_games):
        s, st = play_game(_WORKER["ev"], sims, mc, c_visit, c_scale, rng, max_ply)
        out.extend(s)
        stats.append(st)
    return out, stats


# ------------------------------------------------------------------- config

@dataclass
class Connect4Config:
    channels: int = 48
    blocks: int = 4
    iterations: int = 100000            # runs until stopped; checkpoints every iter
    games_per_iter: int = 48
    n_workers: int = 6                  # <=1 runs self-play inline (no process pool)
    sims: int = 32
    eval_sims: int = 0                  # 0 -> use `sims`; else stronger eval search
    max_considered: int = 16            # = ACTION_SIZE: consider EVERY column at the
                                        # root. With a 16-action game, a narrower set
                                        # (Gumbel-top-k by prior) can exclude a low-
                                        # prior *winning* move so search can never
                                        # play it — and self-play never reinforces it.
    max_ply: int = 64
    c_visit: float = 50.0
    c_scale: float = 0.3
    buffer_size: int = 60000
    batch_size: int = 256
    train_steps: int = 120
    lr: float = 2e-3
    weight_decay: float = 1e-4
    entropy_coef: float = 0.01
    eval_every: int = 5
    eval_games: int = 40
    ckpt_dir: str = "models/connect4"
    seed: int = 0
    device: str = "cpu"
    train_threads: int = 4
    resume: bool = False               # warm-start from ckpt_dir/latest.pt if present


# ------------------------------------------------------------- eval players

class _RandomAgent:
    def __init__(self, rng):
        self.rng = rng

    def select(self, state):
        legal = state.legal_moves()
        return legal[self.rng.randrange(len(legal))]


class _TacticAgent:
    """1-ply tactic: take an immediate win; else block an immediate loss; else random."""

    def __init__(self, rng):
        self.rng = rng

    def select(self, state):
        me = state.to_play
        wins = state.winning_columns(me)
        if wins:
            return wins[0]
        threats = state.winning_columns(-me)
        if threats:
            return threats[0]
        legal = state.legal_moves()
        return legal[self.rng.randrange(len(legal))]


def _greedy_move(evaluator, state, sims, mc, rng):
    a, _ = GumbelMCTS(evaluator, n_sims=sims, max_considered=mc,
                      rng=rng, solve_children=True).run(state, add_noise=False)
    return int(a)


def eval_vs(evaluator, opponent, cfg, rng, seed_offset=0):
    """Play cfg.eval_games vs `opponent`, alternating colors. Returns metrics."""
    eval_sims = cfg.eval_sims or cfg.sims
    wins = draws = losses = 0
    for i in range(cfg.eval_games):
        net_is_p1 = (i % 2 == 0)
        g = Connect4Game()
        while not g.is_terminal() and g.ply < cfg.max_ply:
            net_to_move = (g.to_play == 1) == net_is_p1
            if net_to_move:
                a = _greedy_move(evaluator, g, eval_sims, cfg.max_considered, rng)
            else:
                a = opponent.select(g)
            g = g.apply(a)
        z = g.result()                    # +1's perspective
        net_z = z if net_is_p1 else -z
        wins += net_z > 0
        draws += net_z == 0
        losses += net_z < 0
    n = cfg.eval_games
    return {"score": (wins + 0.5 * draws) / n, "wins": wins, "draws": draws, "losses": losses}


# ---------------------------------------------------------------- train loop

def train(cfg: Connect4Config):
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    torch.set_num_threads(cfg.train_threads)
    rng = np.random.default_rng(cfg.seed)

    ckpt_dir = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    weights_path = ckpt_dir / "_live.pt"
    metrics_path = ckpt_dir / "metrics.jsonl"

    net = Connect4Net(channels=cfg.channels, blocks=cfg.blocks).to(cfg.device)
    # Resume from the last checkpoint in this ckpt_dir if one exists (warm start;
    # optimizer state is reset, which is fine for Adam).
    latest = ckpt_dir / "latest.pt"
    if getattr(cfg, "resume", False) and latest.exists():
        ck = torch.load(latest, map_location=cfg.device, weights_only=False)
        net.load_state_dict(ck["state_dict"])
        print(f"resumed from {latest} (iter {ck.get('iter')})", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    buffer: deque = deque(maxlen=cfg.buffer_size)

    torch.save(net.state_dict(), weights_path)

    use_pool = cfg.n_workers > 1
    pool = None
    if use_pool:
        ctx = mp.get_context("spawn")

        def make_pool():
            return ProcessPoolExecutor(max_workers=cfg.n_workers, mp_context=ctx,
                                       initializer=_init_worker,
                                       initargs=(cfg.channels, cfg.blocks))
        pool = make_pool()
    else:
        inline_ev = Connect4Evaluator(Connect4Net(channels=cfg.channels, blocks=cfg.blocks))

    print(f"connect4 training: {sum(p.numel() for p in net.parameters())/1e6:.3f}M params, "
          f"{'pool x%d' % cfg.n_workers if use_pool else 'inline'}, {cfg.sims} sims, "
          f"device={cfg.device}", flush=True)

    games_done = 0
    try:
        for it in range(1, cfg.iterations + 1):
            t0 = time.time()
            # Self-play always uses a CPU copy of the latest weights.
            cpu_sd = {k: v.detach().cpu() for k, v in net.state_dict().items()}
            torch.save(cpu_sd, weights_path)

            all_stats = []
            if use_pool:
                per = max(1, cfg.games_per_iter // cfg.n_workers)
                tasks = [(str(weights_path), per, cfg.sims, cfg.max_considered,
                          cfg.c_visit, cfg.c_scale, cfg.max_ply,
                          int(rng.integers(1 << 30))) for _ in range(cfg.n_workers)]
                try:
                    for fut in [pool.submit(_worker_play, t) for t in tasks]:
                        samples, stats = fut.result()
                        buffer.extend(samples)
                        all_stats.extend(stats)
                except BrokenProcessPool:
                    print("  pool broke -> recreating", flush=True)
                    pool.shutdown(wait=False, cancel_futures=True)
                    pool = make_pool()
                except Exception as e:
                    print(f"  worker error: {e}", flush=True)
            else:
                inline_ev.net.load_state_dict(cpu_sd)
                for _ in range(cfg.games_per_iter):
                    s, st = play_game(inline_ev, cfg.sims, cfg.max_considered,
                                      cfg.c_visit, cfg.c_scale, rng, cfg.max_ply)
                    buffer.extend(s)
                    all_stats.append(st)
            games_done += len(all_stats)

            # --- training ---
            net.train()
            ploss = vloss = 0.0
            nb = 0
            if len(buffer) >= cfg.batch_size:
                for _ in range(cfg.train_steps):
                    batch = random.sample(buffer, cfg.batch_size)
                    planes = torch.from_numpy(
                        np.stack([b[0] for b in batch]).astype(np.float32)).to(cfg.device)
                    target_pi = torch.zeros(len(batch), ACTION_SIZE)
                    for k, b in enumerate(batch):
                        target_pi[k, b[1].astype(np.int64)] = torch.from_numpy(b[2])
                    target_pi = target_pi.to(cfg.device)
                    target_z = torch.tensor([float(b[3]) for b in batch]).to(cfg.device)

                    logits, value = net(planes)
                    logp = F.log_softmax(logits, dim=1)
                    pl = -(target_pi * logp).sum(dim=1).mean()
                    vl = F.mse_loss(value, target_z)
                    ent = -(logp.exp() * logp).sum(dim=1).mean()
                    loss = pl + vl - cfg.entropy_coef * ent
                    opt.zero_grad(); loss.backward(); opt.step()
                    ploss += pl.item(); vloss += vl.item(); nb += 1
            ploss = ploss / nb if nb else float("nan")
            vloss = vloss / nb if nb else float("nan")

            dec = sum(1 for s in all_stats if s["decisive"])
            avg_plies = np.mean([s["plies"] for s in all_stats]) if all_stats else 0

            rec = {"iter": it, "games": games_done, "buffer": len(buffer),
                   "ploss": round(ploss, 4), "vloss": round(vloss, 4),
                   "selfplay_decisive": dec, "n_games": len(all_stats),
                   "avg_plies": round(float(avg_plies), 1), "secs": round(time.time() - t0, 1)}

            if it % cfg.eval_every == 0:
                net.eval()
                ev = Connect4Evaluator(net, device=cfg.device)
                erng = np.random.default_rng(cfg.seed + it)
                rec["vs_random"] = eval_vs(ev, _RandomAgent(random.Random(1000 + it)), cfg, erng)
                rec["vs_tactic"] = eval_vs(ev, _TacticAgent(random.Random(2000 + it)), cfg, erng)

            line = (f"it {it:4d} | games {games_done:6d} | buf {len(buffer):6d} | "
                    f"ploss {ploss:.3f} vloss {vloss:.3f} | dec {dec}/{len(all_stats)} "
                    f"avg_plies {avg_plies:.0f} | {rec['secs']:.0f}s")
            if "vs_random" in rec:
                line += (f" | vs_random {rec['vs_random']['score']:.2f}"
                         f" | vs_tactic {rec['vs_tactic']['score']:.2f}")
            print(line, flush=True)

            with open(metrics_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

            torch.save({"state_dict": net.state_dict(), "channels": cfg.channels,
                        "blocks": cfg.blocks, "iter": it}, ckpt_dir / "latest.pt")
            if it % 25 == 0:
                torch.save({"state_dict": net.state_dict(), "channels": cfg.channels,
                            "blocks": cfg.blocks, "iter": it}, ckpt_dir / f"iter_{it:05d}.pt")
    finally:
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)


def _config(mode: str) -> Connect4Config:
    if mode == "local":
        # Mac-local solver run (M2, user active): fewer workers, smaller batches.
        return Connect4Config(
            channels=48, blocks=4, iterations=400, resume=False,
            games_per_iter=80, n_workers=5, sims=64, eval_sims=64, max_considered=16,
            buffer_size=200000, batch_size=512, train_steps=200,
            lr=2e-3, entropy_coef=0.01, eval_every=5, eval_games=40,
            device="mps", ckpt_dir="models/connect4_solver", seed=0,
        )
    if mode == "smoke":
        return Connect4Config(channels=32, blocks=3, iterations=12, games_per_iter=24,
                              n_workers=1, sims=64, eval_sims=64, max_considered=16,
                              train_steps=60, batch_size=128, eval_every=3, eval_games=30,
                              ckpt_dir="models/connect4_smoke")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # NOTE on sims: this game's key tactic (blocking an opponent's immediate win)
    # is a 2-ply refutation. FIXED via solve_children in the generic search
    # (az_mcts._expand + gumbel proven-win completion): terminal children are
    # seeded with their exact values at expansion, so a non-blocking move is
    # refuted exactly on its first expansion and a proven win IS the target.
    # Verified on the stuck iter-49 net @48 sims/mc=16: WIN target 0.61->1.00,
    # BLOCK 0.30->0.66 (argmax 3/8->7/8). Low sims are enough by design (Gumbel);
    # sims=64 is deliberate — high sims were a workaround, not the fix.
    return Connect4Config(
        channels=48, blocks=4, iterations=400, resume=False,
        games_per_iter=128, n_workers=48, sims=64, eval_sims=64, max_considered=16,
        buffer_size=200000, batch_size=512, train_steps=200,
        lr=2e-3, entropy_coef=0.01, eval_every=5, eval_games=40,
        device=device, ckpt_dir="models/connect4", seed=0,
    )


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    train(_config(mode))
