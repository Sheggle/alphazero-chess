"""Benchmark: batched Gumbel self-play vs the current per-game engine.

Reports positions/sec (a "position" = one self-play move = one training sample)
and games/min for:

  * the current single-game engine (`chess_train.play_chess_game`), and
  * the batched engine at concurrency B in {1, 8, 32, 64, 128},

on the production 64-channel / 6-block net, under both `torch.set_num_threads(1)`
(the per-worker setting the trainer uses) and default threads. Prints a speedup
table, plus a raw net-forward micro-benchmark (the batching ceiling) and a
single-process-batched vs many-batch-1-workers assessment.

Usage:
    PYTHONPATH=. uv run python scripts/bench_batched.py            # default
    PYTHONPATH=. uv run python scripts/bench_batched.py --quick    # tiny, fast
    PYTHONPATH=. uv run python scripts/bench_batched.py --games 64 --max-ply 40
"""

import argparse
import time

import numpy as np
import torch

from alphazero.chess_net import ChessEvaluator, ChessNet
from alphazero.chess_batched import play_batched_games
from alphazero.chess_train import play_chess_game


def net_micro_bench(net):
    """Raw forward throughput at several batch sizes -- the batching ceiling."""
    net.eval()
    print("\n=== net forward micro-benchmark (the batching ceiling) ===")
    print(f"{'batch':>6} {'ms/call':>9} {'ms/pos':>9} {'pos/sec':>10} {'speedup':>8}")
    base = None
    with torch.no_grad():
        for B in (1, 8, 32, 64, 128):
            x = torch.randn(B, 18, 8, 8)
            for _ in range(3):
                net(x)
            reps = max(5, 200 // B)
            t = time.time()
            for _ in range(reps):
                net(x)
            dt = (time.time() - t) / reps
            per_pos = dt / B
            if base is None:
                base = per_pos
            print(f"{B:>6} {dt*1e3:>9.2f} {per_pos*1e3:>9.3f} "
                  f"{1/per_pos:>10.0f} {base/per_pos:>7.1f}x")


def bench_single(ev, n_games, sims, mc, max_ply, seed):
    rng = np.random.default_rng(seed)
    t = time.time()
    n_pos = 0
    for _ in range(n_games):
        samples, _ = play_chess_game(ev, sims, mc, max_ply, 50.0, 1.0, rng, 1.0)
        n_pos += len(samples)
    dt = time.time() - t
    return n_pos, n_games, dt


def bench_batched(ev, n_games, concurrency, sims, mc, max_ply, seed):
    t = time.time()
    samples, stats = play_batched_games(
        ev, n_games=n_games, concurrency=concurrency, sims=sims,
        max_considered=mc, max_ply=max_ply, seed=seed)
    dt = time.time() - t
    return len(samples), len(stats), dt


def run_suite(threads, n_games, sims, mc, max_ply, seed):
    torch.set_num_threads(threads)
    label = f"torch_threads={threads}"
    print(f"\n=== self-play throughput ({label}), "
          f"{n_games} games, sims={sims}, max_ply={max_ply} ===")

    net = ChessNet(channels=64, blocks=6)
    ev = ChessEvaluator(net)

    # Baseline: current per-game engine.
    pos, gms, dt = bench_single(ev, n_games, sims, mc, max_ply, seed)
    base_pps = pos / dt
    print(f"{'engine':>22} {'pos':>7} {'sec':>7} {'pos/sec':>9} "
          f"{'games/min':>10} {'speedup':>8}")
    print(f"{'single (current)':>22} {pos:>7} {dt:>7.1f} {base_pps:>9.1f} "
          f"{gms/dt*60:>10.1f} {1.0:>7.2f}x")

    for B in (1, 8, 32, 64, 128):
        pos, gms, dt = bench_batched(ev, n_games, B, sims, mc, max_ply, seed)
        pps = pos / dt
        print(f"{('batched B=' + str(B)):>22} {pos:>7} {dt:>7.1f} {pps:>9.1f} "
              f"{gms/dt*60:>10.1f} {pps/base_pps:>7.2f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=32)
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--max-considered", type=int, default=8)
    ap.add_argument("--max-ply", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true",
                    help="tiny config for a fast smoke run")
    args = ap.parse_args()

    if args.quick:
        args.games, args.sims, args.max_ply = 8, 8, 16

    net = ChessNet(channels=64, blocks=6)
    net_micro_bench(net)

    for threads in (1, torch.get_num_threads() if torch.get_num_threads() > 1 else 4):
        run_suite(threads, args.games, args.sims, args.max_considered,
                  args.max_ply, args.seed)

    print("\n=== one big-B process vs many batch-1 worker processes ===")
    print(
        "The current trainer runs ~6-7 OS processes, each doing batch-1 self-play\n"
        "(one torch thread each). A single batched process with large B replaces\n"
        "them: compare the single-process numbers above. Rough guidance:\n"
        "  * total throughput of N batch-1 workers ~= N * (single pos/sec).\n"
        "  * one batched process at B>=32 should match or beat that on CPU and\n"
        "    crush it on GPU, with 1/N the RAM (one net copy, not N) and no\n"
        "    weight-reload/IPC per iteration.\n"
        "  * batched also has near-zero process overhead and keeps the GPU fed,\n"
        "    which batch-1 multiprocessing cannot do (tiny kernels, no overlap).")


if __name__ == "__main__":
    main()
