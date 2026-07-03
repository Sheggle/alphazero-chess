"""Correctness + throughput for the Rust board->planes encode.

1. CORRECTNESS: play random games in lockstep with a python-chess board (moves
   pushed, so ep_square has python-chess "always" semantics, exactly like real
   self-play) and a fastchess board. At every position assert the Rust encode is
   bit-for-bit identical to the reference `encode_board`.
2. THROUGHPUT: Rust single encode/s, Rust batch encode/s, vs Python encode_board/s.
"""
from __future__ import annotations

import sys
import time

import chess
import numpy as np

sys.path.insert(0, "fastchess/pybuild")
import fastchess  # noqa: E402
if not hasattr(fastchess, "Board"):
    import importlib
    sys.modules.pop("fastchess", None)
    fastchess = importlib.import_module("fastchess")

from alphazero.chess_encode import encode_board  # noqa: E402
from alphazero.chess_env import ChessGame  # noqa: E402
from alphazero.chess_env_fast import FastChessGame  # noqa: E402


def correctness(n_games: int, max_ply: int, seed: int):
    rng = np.random.default_rng(seed)
    n_pos = 0
    mismatches = 0
    first_bad = None
    for _ in range(n_games):
        pyb = chess.Board()
        fc = fastchess.Board()
        ply = 0
        while not pyb.is_game_over(claim_draw=True) and ply < max_ply:
            ref = encode_board(ChessGame(pyb))     # reference (python-chess walk)
            got = fc.encode()                      # Rust
            n_pos += 1
            if not np.array_equal(ref, got):
                mismatches += 1
                if first_bad is None:
                    diff = np.argwhere(ref != got)
                    first_bad = (pyb.fen(), diff[:8].tolist())
            moves = list(pyb.legal_moves)
            m = moves[int(rng.integers(len(moves)))]
            pyb.push(m)
            fc.apply_uci(m.uci())
            ply += 1
            # sanity: boards stay in lockstep
            if ply % 17 == 0 and fc.fen() != pyb.fen():
                print(f"  DESYNC at ply {ply}: fc={fc.fen()} py={pyb.fen()}")
        # final (possibly terminal) position too
        ref = encode_board(ChessGame(pyb))
        got = fc.encode()
        n_pos += 1
        if not np.array_equal(ref, got):
            mismatches += 1
            if first_bad is None:
                first_bad = (pyb.fen(), np.argwhere(ref != got)[:8].tolist())
    return n_pos, mismatches, first_bad


def collect_positions(n_target: int, seed: int):
    """Parallel lists of (fastchess board, python-chess ChessGame) positions."""
    rng = np.random.default_rng(seed)
    fcs, cgs = [], []
    while len(fcs) < n_target:
        pyb = chess.Board()
        fc = fastchess.Board()
        ply = 0
        while not pyb.is_game_over(claim_draw=True) and ply < 80 and len(fcs) < n_target:
            fcs.append(fc.clone_board())
            cgs.append(ChessGame(pyb.copy(stack=False)))
            moves = list(pyb.legal_moves)
            m = moves[int(rng.integers(len(moves)))]
            pyb.push(m)
            fc.apply_uci(m.uci())
            ply += 1
    return fcs, cgs


def throughput(fcs, cgs, reps: int):
    # Rust single encode
    t0 = time.perf_counter()
    for _ in range(reps):
        for fc in fcs:
            fc.encode()
    rust_dt = time.perf_counter() - t0
    rust_eps = reps * len(fcs) / rust_dt

    # Rust batch encode (whole list per call)
    t0 = time.perf_counter()
    for _ in range(reps):
        fastchess.encode_batch(fcs)
    batch_dt = time.perf_counter() - t0
    batch_eps = reps * len(fcs) / batch_dt

    # Python reference encode (ChessGame holds the live board: no FEN rebuild)
    py_reps = max(1, reps // 4)  # it's ~20x slower; fewer reps to keep it quick
    t0 = time.perf_counter()
    for _ in range(py_reps):
        for cg in cgs:
            encode_board(cg)
    py_dt = time.perf_counter() - t0
    py_eps = py_reps * len(cgs) / py_dt

    # OLD FastChessGame path: a fresh FastChessGame per node -> encode_board hits
    # the per-node FEN reconstruction (chess.Board(fen)) + piece_map walk. This
    # is the in-context ~235 enc/s pre-optimization bottleneck.
    old_reps = max(1, reps // 20)
    t0 = time.perf_counter()
    for _ in range(old_reps):
        for fc in fcs:
            encode_board(FastChessGame(fc.clone_board()))
    old_dt = time.perf_counter() - t0
    old_eps = old_reps * len(fcs) / old_dt

    return rust_eps, batch_eps, py_eps, old_eps


if __name__ == "__main__":
    if "--no-corr" not in sys.argv:
        print("=== CORRECTNESS (Rust encode vs encode_board) ===", flush=True)
        n_pos, mm, bad = correctness(n_games=600, max_ply=80, seed=1)
        print(f"positions checked: {n_pos}   mismatches: {mm}", flush=True)
        if bad:
            print(f"first mismatch fen={bad[0]} diff_idx={bad[1]}", flush=True)

    print("\n=== THROUGHPUT ===", flush=True)
    fcs, cgs = collect_positions(2000, seed=7)
    rust_eps, batch_eps, py_eps, old_eps = throughput(fcs, cgs, reps=20)
    print(f"OLD FastChessGame (FEN-rebuild) : {old_eps:10.0f} enc/s   (the ~235/s wall)", flush=True)
    print(f"Python encode_board (live board): {py_eps:10.0f} enc/s", flush=True)
    print(f"Rust  encode (1x1)              : {rust_eps:10.0f} enc/s  ({rust_eps/old_eps:.0f}x vs old, {rust_eps/py_eps:.0f}x vs live)", flush=True)
    print(f"Rust  encode_batch              : {batch_eps:10.0f} enc/s  ({batch_eps/old_eps:.0f}x vs old)", flush=True)
