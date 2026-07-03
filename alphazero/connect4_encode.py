"""Encode a 3D-Connect-4 state into network input planes (canonical).

The game is fully symmetric between the two players, so no board mirror is
needed — we only swap "mine" vs "theirs" by whose turn it is. Two planes of
4x4x4:
  0 : side-to-move's discs
  1 : opponent's discs

`encode_state(game) -> np.ndarray` of shape (INPUT_PLANES, 4, 4, 4), float32.
"""

from __future__ import annotations

import numpy as np

from .connect4_env import N

INPUT_PLANES = 2


def encode_state(game) -> np.ndarray:
    board = game.board  # (4,4,4) int8 of {0,+1,-1}
    me = game.to_play
    planes = np.zeros((INPUT_PLANES, N, N, N), dtype=np.float32)
    planes[0] = (board == me)
    planes[1] = (board == -me)
    return planes
