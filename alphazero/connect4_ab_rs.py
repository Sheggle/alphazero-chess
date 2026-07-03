"""Python front-end for the Rust Score-Four alpha-beta engine (fastchess crate,
`connect4.rs`). Same heuristic + search as `connect4_ab.py`, ~100x faster.

The compiled extension is loaded from an absolute path (`alphazero/_rustlib/`)
to avoid the namespace-package collision between the `fastchess/` source crate
directory and the built package when both sit on `sys.path`.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np

from .connect4_ab import DEFAULT_WEIGHTS  # reuse the tuned weights

import sys

_SO = os.path.join(os.path.dirname(__file__), "_rustlib", "fastchess.abi3.so")
# The extension's init symbol is PyInit_fastchess, so the spec must be named
# "fastchess"; loading by absolute path registers it directly (no path scan).
if "fastchess" in sys.modules:
    _mod = sys.modules["fastchess"]
else:
    _spec = importlib.util.spec_from_file_location("fastchess", _SO)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["fastchess"] = _mod
    _spec.loader.exec_module(_mod)

c4_best_move = _mod.c4_best_move
c4_best_move_depth = _mod.c4_best_move_depth
c4_eval = _mod.c4_eval


def _flat(game) -> list[int]:
    # env board is (4,4,4); C-order flatten -> cell = x*16+y*4+z (matches Rust).
    return [int(v) for v in np.asarray(game.board).reshape(-1)]


def best_move(game, time_budget: float = 1.0, weights=DEFAULT_WEIGHTS,
              threads: int = 1):
    """Iterative-deepening search (production path: NMP + LMR + aspiration).

    `threads > 1` = lazy SMP over a shared lock-free TT (native only).
    Scores are floats (1.0 = W3); mate = +/-(100_000 - plies). Returns (col, info).
    """
    w1, w2, _ = weights
    col, depth, nodes, score = c4_best_move(
        _flat(game), int(game.to_play), int(round(time_budget * 1000)),
        float(w1), float(w2), int(threads)
    )
    return col, {"depth": depth, "nodes": nodes, "score": score,
                 "time": time_budget}


def best_move_depth(game, depth: int, weights=DEFAULT_WEIGHTS):
    """Fixed-depth search in EXACT mode (no NMP/LMR): deterministic exact
    alpha-beta to `depth` — the verification/tuning path."""
    w1, w2, _ = weights
    col, d, nodes, score = c4_best_move_depth(
        _flat(game), int(game.to_play), int(depth), float(w1), float(w2)
    )
    return col, {"depth": d, "nodes": nodes, "score": score}


def static_eval(game, weights=DEFAULT_WEIGHTS) -> float:
    """Depth-0 heuristic in the side-to-move frame."""
    w1, w2, _ = weights
    return c4_eval(_flat(game), int(game.to_play), float(w1), float(w2))


class Connect4ABRust:
    """Object wrapper mirroring the Python engine's API (stateless per call)."""

    def __init__(self, weights=DEFAULT_WEIGHTS):
        self.weights = tuple(weights)

    def best_move(self, game, time_budget: float = 1.0, threads: int = 1):
        return best_move(game, time_budget, self.weights, threads)

    def best_move_depth(self, game, depth: int):
        return best_move_depth(game, depth, self.weights)
