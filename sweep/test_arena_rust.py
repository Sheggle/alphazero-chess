"""Bit-exact check for the Rust leaf-parallel PUCT arena (fastchess.arena_match):
at L=1 it must reproduce ordinary sequential PUCT move-for-move.

The reference below is the ONLY Python MCTS in the eval/sweep code, and it exists
SOLELY for this test. It runs `AZMCTS` (sequential PUCT) on `FastChessGame`, which
wraps the SAME Rust Board (identical legal-move ORDER, full-history threefold and
encoding) as the arena — so any residual difference would be a real search bug,
not a board-semantics mismatch. (Comparing against a python-chess-ordered tree
instead would diverge purely on move-ordering tie-breaks.)

Run:  PYTHONPATH=.:fastchess/pybuild python -m sweep.test_arena_rust
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
for _p in (str(ROOT), str(ROOT / "fastchess" / "pybuild")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fastchess  # noqa: E402
from sweep.batched_arena import load_evaluator, make_eval_fn  # noqa: E402

# --- REFERENCE ONLY — not a production path (sequential PUCT, order-matched) ---
from alphazero.chess_env_fast import FastChessGame   # noqa: E402  (Rust-board state)
from alphazero.az_mcts import AZMCTS                  # noqa: E402  (sequential PUCT)


def _ref_game_moves(evW, evB, sims, c_puct=1.5, max_ply=60):
    """REFERENCE ONLY: sequential PUCT (AZMCTS) on the Rust board, argmax visits."""
    g = FastChessGame()
    moves = []
    while not g.is_terminal() and g.ply < max_ply:
        ev = evW if g.to_play == 1 else evB
        vis = AZMCTS(ev, n_sims=sims, c_puct=c_puct).run(g, add_noise=False)
        best_a, best_n = g.legal_moves()[0], -1
        for a in g.legal_moves():
            n = vis.get(a, 0)
            if n > best_n:
                best_n, best_a = n, a
        moves.append(int(best_a))
        g = g.apply(int(best_a))
    return moves


def main():
    runs = ROOT / "sweep" / "runs"
    pairs = [("cfg_001", "cfg_004"), ("cfg_004", "cfg_001"),
             ("cfg_000", "cfg_001"), ("cfg_001", "cfg_001")]
    sims, max_ply = 24, 60
    total = 0
    print("=== Rust arena L=1 vs sequential PUCT (AZMCTS on Rust board) ===")
    for pa, pb in pairs:
        evA = load_evaluator(str(runs / pa / "final.pt"), "cpu")
        evB = load_evaluator(str(runs / pb / "final.pt"), "cpu")
        efA, efB = make_eval_fn(evA, fp16=False), make_eval_fn(evB, fp16=False)
        # Rust arena: n_games=2 (game0 = A=White), L=1, fixed sims, no opening.
        _, st = fastchess.arena_match(efA, efB, 2, 0.0, sims, sims, 1, 1, 1.5, max_ply,
                                      2.0, 0, 0, True)
        rust_moves = list(st["moves"][0])
        ref_moves = _ref_game_moves(evA, evB, sims, max_ply=max_ply)
        ml = min(len(rust_moves), len(ref_moves))
        mism = sum(1 for i in range(ml) if rust_moves[i] != ref_moves[i]) \
            + abs(len(rust_moves) - len(ref_moves))
        total += mism
        print(f"  {pa} vs {pb}: rust={len(rust_moves)} ref={len(ref_moves)} mismatches={mism}")
    ok = total == 0
    print(f"\nTOTAL L=1 move mismatches: {total}")
    print("RESULT:", "PASS (Rust arena L=1 == sequential PUCT, bit-exact)" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
