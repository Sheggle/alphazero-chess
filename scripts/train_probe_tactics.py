"""Standalone training driver that REUSES alphazero.chess_train's parallel
self-play workers but evaluates the tactics suite on the LIVE net every
TACT_EVERY iterations -> a tactics-vs-iter curve from a single run.

Does not modify any source file. Configured by the same env vars as
sweep_chess.py, plus TACT_EVERY (default 3). Appends a curve line to
models/chess_exp/tactics_curves.jsonl and prints per-eval rows.
"""
import json, os, random, time
from collections import deque
from pathlib import Path

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import torch
import torch.nn.functional as F

from alphazero.chess_env import ACTION_SIZE
from alphazero.chess_net import ChessNet, ChessEvaluator
from alphazero.chess_train import (ChessConfig, _init_worker, _worker_play,
                                   eval_vs_random)
from alphazero.chess_tactics import tactics_rates


def _i(k, d): return int(os.environ.get(k, d))
def _f(k, d): return float(os.environ.get(k, d))


def main():
    import sys
    name = sys.argv[1]
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 18
    tact_every = _i("TACT_EVERY", 3)
    expdir = Path("models/chess_exp") / name
    cfg = ChessConfig(
        channels=_i("CH", 32), blocks=_i("BL", 4), iterations=iters,
        games_per_iter=_i("GPI", 14), n_workers=_i("NW", 3),
        sims=_i("SIMS", 24), max_considered=_i("MC", 8), max_ply=_i("MAXPLY", 60),
        c_visit=_f("CVISIT", 50.0), c_scale=_f("CSCALE", 1.0),
        mat_thresh=_f("MATTHRESH", 1.0), buffer_size=_i("BUF", 60000),
        batch_size=_i("BATCH", 256), train_steps=_i("TS", 100), lr=_f("LR", 2e-3),
        weight_decay=_f("WD", 1e-4), entropy_coef=_f("ENT", 0.01),
        eval_every=10**9, eval_games=_i("EVAL_GAMES", 14),
        ckpt_dir=str(expdir), seed=_i("SEED", 0), train_threads=_i("TT", 2),
    )
    knobs = dict(CH=cfg.channels, BL=cfg.blocks, SIMS=cfg.sims, MC=cfg.max_considered,
                 LR=cfg.lr, ENT=cfg.entropy_coef, CSCALE=cfg.c_scale,
                 CVISIT=cfg.c_visit, TS=cfg.train_steps, GPI=cfg.games_per_iter)
    print(f"=== {name}: {knobs} ===", flush=True)

    torch.manual_seed(cfg.seed); random.seed(cfg.seed)
    torch.set_num_threads(cfg.train_threads)
    rng = np.random.default_rng(cfg.seed)
    expdir.mkdir(parents=True, exist_ok=True)
    weights_path = expdir / "_live.pt"

    net = ChessNet(channels=cfg.channels, blocks=cfg.blocks)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    buffer = deque(maxlen=cfg.buffer_size)
    torch.save(net.state_dict(), weights_path)
    ctx = mp.get_context("spawn")

    def make_pool():
        return ProcessPoolExecutor(max_workers=cfg.n_workers, mp_context=ctx,
            initializer=_init_worker, initargs=(cfg.channels, cfg.blocks))
    pool = make_pool()

    # fresh-net tactics baseline (at this MC/SIMS)
    net.eval()
    base = tactics_rates(ChessEvaluator(net), sims=cfg.sims, max_considered=cfg.max_considered)
    curve = [{"iter": 0, "tactics": base}]
    print(f"  iter  0 | fresh | tac overall {base['overall']:.3f} "
          f"(m1 {base['mate_in_1']:.2f} hc {base['hanging_capture']:.2f})", flush=True)

    t_start = time.time()
    try:
        for it in range(1, iters + 1):
            t0 = time.time()
            torch.save(net.state_dict(), weights_path)
            per = max(1, cfg.games_per_iter // cfg.n_workers)
            tasks = [(str(weights_path), per, cfg.sims, cfg.max_considered, cfg.max_ply,
                      cfg.c_visit, cfg.c_scale, cfg.mat_thresh,
                      int(rng.integers(1 << 30))) for _ in range(cfg.n_workers)]
            all_stats = []
            try:
                for fut in [pool.submit(_worker_play, t) for t in tasks]:
                    s, st = fut.result(); buffer.extend(s); all_stats.extend(st)
            except BrokenProcessPool:
                pool.shutdown(wait=False, cancel_futures=True); pool = make_pool()

            net.train(); ploss = vloss = 0.0; nb = 0
            if len(buffer) >= cfg.batch_size:
                for _ in range(cfg.train_steps):
                    batch = random.sample(buffer, cfg.batch_size)
                    planes = torch.from_numpy(np.stack([b[0] for b in batch]).astype(np.float32))
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
            ploss = ploss/nb if nb else float("nan"); vloss = vloss/nb if nb else float("nan")
            avg_plies = float(np.mean([s["plies"] for s in all_stats])) if all_stats else 0
            secs = time.time()-t0

            row = f"  iter {it:2d} | ploss {ploss:.3f} vloss {vloss:.3f} avgply {avg_plies:.0f} | {secs:.0f}s"
            if it % tact_every == 0 or it == iters:
                net.eval()
                tac = tactics_rates(ChessEvaluator(net), sims=cfg.sims, max_considered=cfg.max_considered)
                curve.append({"iter": it, "tactics": tac, "vloss": round(vloss,4), "ploss": round(ploss,4)})
                row += (f" | TAC overall {tac['overall']:.3f} "
                        f"(m1 {tac['mate_in_1']:.2f} hc {tac['hanging_capture']:.2f})")
            print(row, flush=True)
            torch.save({"state_dict": net.state_dict(), "channels": cfg.channels,
                        "blocks": cfg.blocks, "iter": it}, expdir / "latest.pt")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    wall = time.time() - t_start
    net.eval()
    vr = eval_vs_random(ChessEvaluator(net), cfg, np.random.default_rng(999))
    out = {"name": name, "iters": iters, "wall_s": round(wall,1), "knobs": knobs,
           "curve": curve, "vs_random": vr}
    with open("models/chess_exp/tactics_curves.jsonl", "a") as f:
        f.write(json.dumps(out) + "\n")
    print(f"=== {name} DONE {wall:.0f}s | fresh {base['overall']:.3f} -> "
          f"final {curve[-1]['tactics']['overall']:.3f} | vs_random {vr['score']:.2f} ===", flush=True)


if __name__ == "__main__":
    main()
