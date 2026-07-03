"""Fast proxy experiment for chess hyperparameter tuning.

Usage: sweep_chess.py <name> <iters>   (knobs via env vars)
Runs a bounded proxy training, then a final tactics + vs-random eval, appends a
result line to models/chess_exp/results.jsonl, and writes a DONE marker.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from alphazero.chess_train import ChessConfig, eval_vs_random, train
from alphazero.chess_net import ChessEvaluator, ChessNet


def _i(k, d): return int(os.environ.get(k, d))
def _f(k, d): return float(os.environ.get(k, d))


def main():
    name = sys.argv[1]
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    expdir = Path("models/chess_exp") / name
    cfg = ChessConfig(
        channels=_i("CH", 32), blocks=_i("BL", 4),
        iterations=iters, games_per_iter=_i("GPI", 14), n_workers=_i("NW", 7),
        sims=_i("SIMS", 24), max_considered=_i("MC", 8), max_ply=_i("MAXPLY", 60),
        c_visit=_f("CVISIT", 50.0), c_scale=_f("CSCALE", 1.0),
        mat_thresh=_f("MATTHRESH", 1.0),
        buffer_size=_i("BUF", 60000), batch_size=_i("BATCH", 256),
        train_steps=_i("TS", 100), lr=_f("LR", 2e-3),
        weight_decay=_f("WD", 1e-4), entropy_coef=_f("ENT", 0.01),
        eval_every=_i("EVAL_EVERY", 5), eval_games=_i("EVAL_GAMES", 14),
        ckpt_dir=str(expdir), seed=_i("SEED", 0), train_threads=_i("TT", 4),
    )
    knobs = {k: getattr(cfg, k) for k in ("channels", "blocks", "sims", "max_considered",
             "max_ply", "lr", "entropy_coef", "train_steps", "games_per_iter",
             "mat_thresh", "c_scale", "batch_size")}
    print(f"=== {name}: {knobs} ===", flush=True)
    t0 = time.time()
    train(cfg)
    wall = time.time() - t0

    # final eval on the trained net
    ck = torch.load(expdir / "latest.pt", map_location="cpu", weights_only=False)
    net = ChessNet(channels=ck["channels"], blocks=ck["blocks"]); net.load_state_dict(ck["state_dict"])
    ev = ChessEvaluator(net)
    vr = eval_vs_random(ev, cfg, np.random.default_rng(999))
    tactics = None
    try:
        from alphazero.chess_tactics import tactics_rates
        tactics = tactics_rates(ev, sims=cfg.sims, max_considered=cfg.max_considered)
    except Exception as e:
        print(f"(tactics unavailable: {e})", flush=True)

    # pull last training metrics
    last = {}
    mp = expdir / "metrics.jsonl"
    if mp.exists():
        lines = mp.read_text().strip().splitlines()
        if lines:
            last = json.loads(lines[-1])
    result = {"name": name, "iters": iters, "wall_s": round(wall, 1),
              "knobs": knobs, "ploss": last.get("ploss"), "vloss": last.get("vloss"),
              "vs_random": vr, "tactics": tactics,
              "avg_plies": last.get("avg_plies")}
    Path("models/chess_exp").mkdir(parents=True, exist_ok=True)
    with open("models/chess_exp/results.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")
    print("RESULT", json.dumps(result), flush=True)
    (expdir / "DONE").write_text("done")
    print(f"=== {name} DONE in {wall:.0f}s ===", flush=True)


if __name__ == "__main__":
    main()
