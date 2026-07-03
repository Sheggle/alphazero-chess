"""Parallel self-play throughput benchmark.

Mirrors the 1.6 games/s baseline settings: 64ch/6b net, sims=32,
max_considered=16, max_ply=80. Each worker runs real `play_chess_game` (same
GumbelMCTS, same hyperparams) until a wall-time budget, then reports games and
positions (plies) produced. Switch between the Rust-backed FastChessGame and the
python-chess ChessGame with --baseline.

Usage: python bench_selfplay.py <n_procs> <duration_s> [--baseline]
"""
from __future__ import annotations

import multiprocessing as mp
import sys
import time

import numpy as np
import torch

from alphazero.chess_env import ChessGame
from alphazero.chess_env_fast import FastChessGame
from alphazero.chess_net import ChessEvaluator, ChessNet
from alphazero.chess_train import play_chess_game

SIMS = 32
MAX_CONSIDERED = 16
MAX_PLY = 80
C_VISIT = 50.0
C_SCALE = 1.0
CHANNELS = 64
BLOCKS = 6


def worker(args):
    seed, duration, mode = args
    torch.set_num_threads(1)
    if mode == "oldfast":
        # Reproduce the pre-optimization path: FastChessGame board mechanics but
        # encode via python-chess (`encode_board` -> g.board FEN reconstruction
        # + piece_map walk). This isolates exactly the encode win. Patch BOTH the
        # evaluator's bound name (the per-simulation hot path) and chess_train's.
        from alphazero.chess_encode import encode_board
        import alphazero.chess_net as cn
        import alphazero.chess_train as ct
        cn.encode_state = encode_board
        ct.encode_state = encode_board
    net = ChessNet(channels=CHANNELS, blocks=BLOCKS)
    net.eval()
    ev = ChessEvaluator(net)
    rng = np.random.default_rng(seed)
    game_cls = ChessGame if mode == "baseline" else FastChessGame
    games = plies = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < duration:
        samples, stats = play_chess_game(
            ev, SIMS, MAX_CONSIDERED, MAX_PLY, C_VISIT, C_SCALE, rng,
            mat_thresh=1.0, game_cls=game_cls)
        games += 1
        plies += stats["plies"]
    elapsed = time.perf_counter() - t0
    return games, plies, elapsed


def main():
    n_procs = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    if "--baseline" in sys.argv:
        mode = "baseline"
    elif "--oldfast" in sys.argv:
        mode = "oldfast"
    else:
        mode = "fast"
    label = {"baseline": "BASELINE (python-chess ChessGame)",
             "oldfast": "OLDFAST (FastChessGame + python-chess encode_board)",
             "fast": "FAST (Rust FastChessGame + Rust encode)"}[mode]

    ctx = mp.get_context("fork")
    args = [(1000 + i, duration, mode) for i in range(n_procs)]
    t0 = time.perf_counter()
    with ctx.Pool(n_procs) as pool:
        results = pool.map(worker, args)
    wall = time.perf_counter() - t0

    total_games = sum(r[0] for r in results)
    total_plies = sum(r[1] for r in results)
    mean_elapsed = np.mean([r[2] for r in results])

    gps = total_games / mean_elapsed
    pps = total_plies / mean_elapsed
    print(f"\n=== {label} | {n_procs} procs | {duration:.0f}s/proc ===")
    print(f"games: {total_games}  plies: {total_plies}  mean_worker_elapsed: {mean_elapsed:.1f}s  wall: {wall:.1f}s")
    print(f"throughput: {gps:.2f} games/s   {pps:.0f} pos/s (plies/s)")
    print(f"per-proc:   {gps/n_procs:.3f} games/s   {pps/n_procs:.1f} pos/s")


if __name__ == "__main__":
    main()
