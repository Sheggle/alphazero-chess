"""Tic-tac-toe game logic.

State is immutable: every move returns a fresh state. The board is a length-9
tuple of int8 in {+1, -1, 0}: +1 is the first player (X), -1 the second (O),
0 empty. Cells are indexed row-major:

    0 | 1 | 2
    ---------
    3 | 4 | 5
    ---------
    6 | 7 | 8

`to_play` is +1 or -1 — whose turn it is. We keep a single canonical board
(absolute, not perspective-flipped); perspective handling lives in the encoder
used by the network, so the raw game stays easy to reason about.
"""

from __future__ import annotations

from dataclasses import dataclass

# All 8 lines that win the game.
WIN_LINES: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
    (0, 4, 8), (2, 4, 6),             # diagonals
)


@dataclass(frozen=True, slots=True)
class TicTacToe:
    board: tuple[int, ...] = (0,) * 9
    to_play: int = 1  # +1 (X) moves first

    # --- generic game interface (shared by every game we'll add later) ---

    @property
    def action_size(self) -> int:
        return 9

    def legal_moves(self) -> list[int]:
        return [i for i, v in enumerate(self.board) if v == 0]

    def legal_mask(self) -> tuple[bool, ...]:
        return tuple(v == 0 for v in self.board)

    def apply(self, action: int) -> "TicTacToe":
        if self.board[action] != 0:
            raise ValueError(f"illegal move {action} on board {self.board}")
        new_board = list(self.board)
        new_board[action] = self.to_play
        return TicTacToe(tuple(new_board), -self.to_play)

    def winner(self) -> int:
        """+1 / -1 if that player has three in a row, else 0."""
        b = self.board
        for a, c, d in WIN_LINES:
            s = b[a] + b[c] + b[d]
            if s == 3:
                return 1
            if s == -3:
                return -1
        return 0

    def is_terminal(self) -> bool:
        return self.winner() != 0 or all(v != 0 for v in self.board)

    def result(self) -> int:
        """Game outcome from player +1's perspective: +1 win, -1 loss, 0 draw.

        Only meaningful on a terminal state.
        """
        return self.winner()

    # --- convenience ---

    def __str__(self) -> str:
        sym = {1: "X", -1: "O", 0: "."}
        rows = [
            " ".join(sym[self.board[r * 3 + c]] for c in range(3))
            for r in range(3)
        ]
        return "\n".join(rows)
