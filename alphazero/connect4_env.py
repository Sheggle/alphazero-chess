"""3D Connect 4 ("Score Four", 4x4x4) in the generic game interface.

Wraps a 4x4x4 cube in exactly the interface the generic search consumes
(`to_play`, `action_size`, `legal_moves`, `apply`, `is_terminal`, `result`),
mirroring `ChessGame`. The state is immutable / copy-on-apply.

Geometry
--------
16 columns indexed by (x, y), x, y in 0..3. A MOVE is a column choice; the disc
falls to the lowest empty height z (gravity). So ACTION_SIZE = 16 (z is implied,
never chosen). A column is full when all 4 heights are used.

Win = 4 of one player's discs in any straight line in the cube: the 3 axes, the
2D face diagonals, and the 4 main space diagonals — 76 lines total, precomputed
once below.

Perspective: the board stores absolute disc values (+1 for player +1, -1 for
player -1). `result()` returns the winner in player +1's frame (+1/-1/0), and
`to_play` is +1/-1 — matching the negamax convention the search expects
(terminal value = `result() * to_play`).
"""

from __future__ import annotations

import numpy as np

N = 4
ACTION_SIZE = 16  # one action per column; z is chosen by gravity


def _cell(x: int, y: int, z: int) -> int:
    """Flatten (x,y,z) to a 0..63 cell index."""
    return x * 16 + y * 4 + z


def _gen_lines() -> list[tuple[tuple[int, int, int], ...]]:
    """All 76 winning lines as tuples of four (x, y, z) cells."""
    dirs = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]
    seen: set[frozenset] = set()
    lines: list[tuple[tuple[int, int, int], ...]] = []
    for dx, dy, dz in dirs:
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    # Only start a line where it has no in-bounds predecessor,
                    # so each geometric line is produced once per direction; the
                    # frozenset guard then dedupes the two opposite directions.
                    px, py, pz = x - dx, y - dy, z - dz
                    if 0 <= px < N and 0 <= py < N and 0 <= pz < N:
                        continue
                    ex, ey, ez = x + 3 * dx, y + 3 * dy, z + 3 * dz
                    if not (0 <= ex < N and 0 <= ey < N and 0 <= ez < N):
                        continue
                    cells = tuple(
                        (x + i * dx, y + i * dy, z + i * dz) for i in range(N)
                    )
                    key = frozenset(cells)
                    if key in seen:
                        continue
                    seen.add(key)
                    lines.append(cells)
    return lines


LINES = _gen_lines()
assert len(LINES) == 76, f"expected 76 win lines, got {len(LINES)}"

# Flattened line cell indices, and, per cell, the lines passing through it.
_LINE_IDX = [tuple(_cell(*c) for c in line) for line in LINES]
_LINES_THROUGH: list[list[tuple[int, int, int, int]]] = [[] for _ in range(64)]
for _li in _LINE_IDX:
    for _c in _li:
        _LINES_THROUGH[_c].append(_li)


class Connect4Game:
    action_size = ACTION_SIZE

    __slots__ = ("board", "_to_play", "_winner", "_ply", "_legal")

    def __init__(self, board: np.ndarray | None = None, to_play: int = 1,
                 winner: int = 0, ply: int = 0):
        # board: (4,4,4) int8 of {0, +1, -1}; board[x, y, z].
        self.board = board if board is not None else np.zeros((N, N, N), dtype=np.int8)
        self._to_play = to_play
        self._winner = winner
        self._ply = ply
        self._legal: list[int] | None = None

    # --- generic game interface ---

    @property
    def to_play(self) -> int:
        return self._to_play

    @property
    def ply(self) -> int:
        return self._ply

    def _drop_z(self, x: int, y: int) -> int:
        """Lowest empty height in column (x, y), or -1 if the column is full."""
        col = self.board[x, y]
        for z in range(N):
            if col[z] == 0:
                return z
        return -1

    def legal_moves(self) -> list[int]:
        if self._legal is None:
            # A column is playable iff its top cell (z=3) is still empty.
            self._legal = [c for c in range(ACTION_SIZE)
                           if self.board[c // N, c % N, N - 1] == 0]
        return self._legal

    def apply(self, action: int) -> "Connect4Game":
        x, y = action // N, action % N
        z = self._drop_z(x, y)
        if z < 0:
            raise ValueError(f"column {action} is full / illegal")
        b = self.board.copy()
        b[x, y, z] = self._to_play
        winner = self._to_play if self._is_win(b, x, y, z, self._to_play) else 0
        return Connect4Game(b, -self._to_play, winner, self._ply + 1)

    def is_terminal(self) -> bool:
        return self._winner != 0 or len(self.legal_moves()) == 0

    def result(self) -> int:
        """Winner in player +1's perspective: +1, -1, or 0 (draw / non-terminal)."""
        return int(self._winner)

    # --- helpers (used by tactic baseline + win detection) ---

    @staticmethod
    def _is_win(board: np.ndarray, x: int, y: int, z: int, player: int) -> bool:
        """Did placing `player` at (x, y, z) complete a line through that cell?"""
        flat = board.reshape(-1)
        for line in _LINES_THROUGH[_cell(x, y, z)]:
            if all(flat[c] == player for c in line):
                return True
        return False

    def winning_columns(self, player: int) -> list[int]:
        """Columns where `player` dropping now immediately completes a line."""
        out = []
        flat = self.board.reshape(-1)
        for c in self.legal_moves():
            x, y = c // N, c % N
            z = self._drop_z(x, y)
            cell = _cell(x, y, z)
            for line in _LINES_THROUGH[cell]:
                # this cell would become `player`; need the other 3 already player
                if all(flat[k] == player for k in line if k != cell):
                    out.append(c)
                    break
        return out

    def __str__(self) -> str:
        sym = {0: ".", 1: "X", -1: "O"}
        rows = []
        for z in range(N - 1, -1, -1):
            line = "  ".join(
                " ".join(sym[int(self.board[x, y, z])] for y in range(N))
                for x in range(N)
            )
            rows.append(f"z={z}  {line}")
        return "\n".join(rows)
