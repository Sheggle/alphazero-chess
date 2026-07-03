"""Encode a game state into network input.

AlphaZero feeds the board *from the perspective of the player to move*, so the
network never needs to know whose turn it is — it always plays "me vs them".
For tic-tac-toe we use two 3x3 planes:

    plane 0: cells owned by the player to move   (1.0 / 0.0)
    plane 1: cells owned by the opponent         (1.0 / 0.0)

So the network's value head predicts "how good is this position for the side to
move", in [-1, 1], and the policy head is a distribution over the 9 cells.
"""

from __future__ import annotations

import numpy as np

INPUT_PLANES = 2
BOARD = 9


def encode(state) -> np.ndarray:
    """Return a (2, 3, 3) float32 tensor from the side-to-move's perspective."""
    me = state.to_play
    board = np.asarray(state.board, dtype=np.int8).reshape(3, 3)
    planes = np.zeros((INPUT_PLANES, 3, 3), dtype=np.float32)
    planes[0] = (board == me)
    planes[1] = (board == -me)
    return planes


def _transform(grid: np.ndarray, k: int, flip: bool) -> np.ndarray:
    g = np.rot90(grid, k)
    return np.fliplr(g) if flip else g


def symmetries(planes: np.ndarray, pi: np.ndarray):
    """Yield the 8 D4 symmetries of (planes, policy).

    `planes` is (2,3,3), `pi` is (9,). Each board symmetry permutes the cells
    identically for the input planes and the per-cell policy, so the augmented
    samples are all valid training targets. Deduplicated for symmetric boards.
    """
    pi_grid = pi.reshape(3, 3)
    seen = set()
    for flip in (False, True):
        for k in range(4):
            p = np.stack([_transform(planes[c], k, flip) for c in range(planes.shape[0])])
            pg = _transform(pi_grid, k, flip)
            key = (p.tobytes(), pg.tobytes())
            if key in seen:
                continue
            seen.add(key)
            yield p.copy(), pg.reshape(-1).copy()
