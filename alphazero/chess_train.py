"""AlphaZero chess training: parallel Gumbel self-play + central training.

Designed to run overnight on a Mac CPU and survive crashes/sleep:
  - self-play runs in N worker processes (1 torch thread each) for throughput;
  - the trainer holds the net/optimizer/replay buffer centrally;
  - every iteration checkpoints weights + appends a metrics line to disk.

Pragmatic choices for a from-scratch CPU run (weak players rarely checkmate):
  - games are capped at `max_ply`; capped games are adjudicated by material
    (side up >= `mat_thresh` points "wins") so the value head gets a real signal
    and the net quickly learns material — bootstrapping visible chess play.
  - low-sim Gumbel self-play (our low-sim work): strong at ~32 sims, cheap.
"""

from __future__ import annotations

import json
import random
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import chess
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import torch
import torch.nn.functional as F

from .chess_encode import encode_board, encode_state
from .chess_env import ACTION_SIZE, ChessGame
from .chess_net import ChessEvaluator, ChessNet
from .gumbel import GumbelMCTS

_VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def material_diff(board: chess.Board) -> int:
    d = 0
    for _, p in board.piece_map().items():
        v = _VAL.get(p.piece_type, 0)
        d += v if p.color == chess.WHITE else -v
    return d  # White's perspective


def _mat_diff(game) -> int:
    """White-perspective material balance. Prefers the game's own fast Rust
    implementation (FastChessGame) over walking python-chess's piece_map."""
    m = getattr(game, "material_diff", None)
    return m() if m is not None else material_diff(game.board)


def _outcome_white(game, mat_thresh: float) -> int:
    if game.is_terminal():
        return game.result()
    d = _mat_diff(game)
    return 1 if d >= mat_thresh else (-1 if d <= -mat_thresh else 0)


def play_chess_game(evaluator, sims, mc, max_ply, c_visit, c_scale, rng,
                    mat_thresh=1.0, game_cls=ChessGame):
    """One self-play game. Returns (samples, stats). Samples are sparse:
    (planes float16, pi_indices int16, pi_values float32, z float32).

    Value target: real terminations (checkmate/stalemate/draw) use the true
    result (+/-1/0). Capped games — the common case for weak players — are
    anchored to IMMEDIATE material `tanh(material_stm/5)` plus a smaller bonus for
    the eventual material-adjudicated outcome. The material anchor is the key fix:
    it gives a clean, low-noise, correctly-signed value signal so the value head
    learns "more material = better" and steers the search to win material (instead
    of overfitting noisy capped-game outcomes — diagnosed value r<0 problem).
    """
    g = game_cls()
    recs = []
    while not g.is_terminal() and g.ply < max_ply:
        a, pi = GumbelMCTS(evaluator, n_sims=sims, max_considered=mc,
                           c_visit=c_visit, c_scale=c_scale, rng=rng).run(g, add_noise=True)
        recs.append((encode_state(g), pi, g.to_play, _mat_diff(g)))
        g = g.apply(int(a))

    terminal = g.is_terminal()
    z_white = _outcome_white(g, mat_thresh)  # +1/-1/0, White perspective
    samples = []
    for planes, pi, to_play, mat_w in recs:
        if terminal:
            z = float(g.result() * to_play)
        else:
            mat_stm = mat_w * to_play
            z = float(np.clip(np.tanh(mat_stm / 5.0) + 0.5 * (z_white * to_play), -1.0, 1.0))
        idx = np.nonzero(pi)[0].astype(np.int16)
        samples.append((planes.astype(np.float16), idx, pi[idx].astype(np.float32), np.float32(z)))
    stats = {"terminal": terminal, "plies": g.ply, "z_white": z_white,
             "result": g.result() if terminal else None}
    return samples, stats


# ---- worker process plumbing (spawn-safe; functions are module-level) ----

_WORKER = {}


def _init_worker(channels, blocks):
    torch.set_num_threads(1)
    net = ChessNet(channels=channels, blocks=blocks)
    _WORKER["net"] = net
    _WORKER["ev"] = ChessEvaluator(net)


def _worker_play(args):
    (weights_path, n_games, sims, mc, max_ply, c_visit, c_scale, mat_thresh, seed) = args
    sd = torch.load(weights_path, map_location="cpu", weights_only=True)
    _WORKER["net"].load_state_dict(sd)
    rng = np.random.default_rng(seed)
    out, stats = [], []
    for _ in range(n_games):
        s, st = play_chess_game(_WORKER["ev"], sims, mc, max_ply, c_visit, c_scale, rng, mat_thresh)
        out.extend(s)
        stats.append(st)
    return out, stats


@dataclass
class ChessConfig:
    channels: int = 64
    blocks: int = 6
    iterations: int = 100000           # effectively "until stopped"
    games_per_iter: int = 24
    n_workers: int = 6
    sims: int = 32
    max_considered: int = 8
    max_ply: int = 100
    c_visit: float = 50.0
    c_scale: float = 1.0
    mat_thresh: float = 1.0
    buffer_size: int = 150000
    batch_size: int = 256
    train_steps: int = 200
    lr: float = 2e-3
    weight_decay: float = 1e-4
    entropy_coef: float = 0.01
    eval_every: int = 10
    eval_games: int = 20
    ckpt_dir: str = "models/chess"
    seed: int = 0
    train_threads: int = 4


class _RandomAgent:
    def __init__(self, rng):
        self.rng = rng

    def select(self, state):
        legal = state.legal_moves()
        return legal[self.rng.randrange(len(legal))]


def _greedy_chess_move(evaluator, state, sims, mc, rng):
    a, _ = GumbelMCTS(evaluator, n_sims=sims, max_considered=mc,
                      rng=rng).run(state, add_noise=False)
    return int(a)


def eval_vs_random(evaluator, cfg, rng):
    """Play eval_games vs a random mover, alternating colors. Capped games are
    material-adjudicated. Returns dict of metrics."""
    wins = draws = losses = 0
    lengths = []
    rand = _RandomAgent(random.Random(12345))
    for i in range(cfg.eval_games):
        net_is_white = (i % 2 == 0)
        g = ChessGame()
        while not g.is_terminal() and g.ply < cfg.max_ply:
            if (g.to_play == 1) == net_is_white:
                a = _greedy_chess_move(evaluator, g, cfg.sims, cfg.max_considered, rng)
            else:
                a = rand.select(g)
            g = g.apply(a)
        zc = _outcome_white(g, cfg.mat_thresh)
        net_z = zc if net_is_white else -zc
        wins += net_z > 0
        draws += net_z == 0
        losses += net_z < 0
        lengths.append(g.ply)
    n = cfg.eval_games
    return {"score": (wins + 0.5 * draws) / n, "wins": wins, "draws": draws,
            "losses": losses, "avg_len": sum(lengths) / n}


def train(cfg: ChessConfig):
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    torch.set_num_threads(cfg.train_threads)
    rng = np.random.default_rng(cfg.seed)

    ckpt_dir = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    weights_path = ckpt_dir / "_live.pt"
    metrics_path = ckpt_dir / "metrics.jsonl"

    net = ChessNet(channels=cfg.channels, blocks=cfg.blocks)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    buffer: deque = deque(maxlen=cfg.buffer_size)

    torch.save(net.state_dict(), weights_path)
    ctx = mp.get_context("spawn")

    def make_pool():
        return ProcessPoolExecutor(max_workers=cfg.n_workers, mp_context=ctx,
                                   initializer=_init_worker, initargs=(cfg.channels, cfg.blocks))

    pool = make_pool()

    print(f"chess training: {sum(p.numel() for p in net.parameters())/1e6:.1f}M params, "
          f"{cfg.n_workers} workers, {cfg.sims} sims, cap {cfg.max_ply} plies", flush=True)

    games_done = 0
    try:
        for it in range(1, cfg.iterations + 1):
            t0 = time.time()
            torch.save(net.state_dict(), weights_path)

            # --- parallel self-play ---
            per = max(1, cfg.games_per_iter // cfg.n_workers)
            tasks = [(str(weights_path), per, cfg.sims, cfg.max_considered, cfg.max_ply,
                      cfg.c_visit, cfg.c_scale, cfg.mat_thresh,
                      int(rng.integers(1 << 30))) for _ in range(cfg.n_workers)]
            all_stats = []
            try:
                for fut in [pool.submit(_worker_play, t) for t in tasks]:
                    samples, stats = fut.result()
                    buffer.extend(samples)
                    all_stats.extend(stats)
            except BrokenProcessPool:
                print("  pool broke -> recreating", flush=True)
                pool.shutdown(wait=False, cancel_futures=True)
                pool = make_pool()
            except Exception as e:  # keep the night alive
                print(f"  worker error: {e}", flush=True)
            games_done += len(all_stats)

            # --- training ---
            net.train()
            ploss = vloss = 0.0
            nb = 0
            if len(buffer) >= cfg.batch_size:
                for _ in range(cfg.train_steps):
                    batch = random.sample(buffer, cfg.batch_size)
                    planes = torch.from_numpy(
                        np.stack([b[0] for b in batch]).astype(np.float32))
                    target_pi = torch.zeros(len(batch), ACTION_SIZE)
                    for k, b in enumerate(batch):
                        target_pi[k, b[1].astype(np.int64)] = torch.from_numpy(b[2])
                    target_z = torch.tensor([float(b[3]) for b in batch])

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

            # --- decisive fraction from self-play ---
            dec = sum(1 for s in all_stats if s["z_white"] != 0)
            term = sum(1 for s in all_stats if s["terminal"])
            avg_plies = np.mean([s["plies"] for s in all_stats]) if all_stats else 0

            rec = {"iter": it, "games": games_done, "buffer": len(buffer),
                   "ploss": round(ploss, 4), "vloss": round(vloss, 4),
                   "selfplay_decisive": dec, "selfplay_terminal": term,
                   "avg_plies": round(float(avg_plies), 1), "secs": round(time.time() - t0, 1)}

            if it % cfg.eval_every == 0:
                net.eval()
                ev = ChessEvaluator(net)
                rec["vs_random"] = eval_vs_random(ev, cfg, np.random.default_rng(cfg.seed + it))
                try:
                    from .chess_tactics import tactics_rates
                    rec["tactics"] = tactics_rates(ev, sims=cfg.sims, max_considered=cfg.max_considered)
                except Exception:
                    rec["tactics"] = None
                # value-vs-material correlation: the key learning-health metric.
                vrng = random.Random(cfg.seed + it)
                vs_, ms_ = [], []
                for _ in range(80):
                    b = chess.Board()
                    for _ in range(vrng.randint(6, 40)):
                        if b.is_game_over():
                            break
                        b.push(vrng.choice(list(b.legal_moves)))
                    if b.is_game_over():
                        continue
                    _, vv = ev.predict(ChessGame(b))
                    vs_.append(vv)
                    ms_.append(material_diff(b) * (1 if b.turn == chess.WHITE else -1))
                rec["value_r"] = float(np.corrcoef(vs_, ms_)[0, 1]) if len(vs_) > 4 else None

            print(
                f"it {it:4d} | games {games_done:6d} | buf {len(buffer):6d} | "
                f"ploss {ploss:.3f} vloss {vloss:.3f} | dec {dec}/{len(all_stats)} "
                f"avg_plies {avg_plies:.0f} | {rec['secs']:.0f}s"
                + (f" | vs_rand {rec['vs_random']['score']:.2f}"
                   f" | tactics {rec.get('tactics') and round(rec['tactics']['overall'],2)}"
                   f" | val_r {rec.get('value_r') and round(rec['value_r'],2)}"
                   if "vs_random" in rec else ""),
                flush=True)

            with open(metrics_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

            # checkpoint every iteration (cheap insurance)
            torch.save({"state_dict": net.state_dict(), "channels": cfg.channels,
                        "blocks": cfg.blocks, "iter": it}, ckpt_dir / "latest.pt")
            if it % 25 == 0:
                torch.save({"state_dict": net.state_dict(), "channels": cfg.channels,
                            "blocks": cfg.blocks, "iter": it}, ckpt_dir / f"iter_{it:05d}.pt")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    train(ChessConfig())
