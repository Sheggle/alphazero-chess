"""Chess game wrapper + AlphaZero move encoding.

Wraps python-chess in the generic game interface the search expects
(`to_play`, `legal_moves`, `apply`, `is_terminal`, `result`, `action_size`).

Everything is **canonical**: the board is always seen from the side to move
("I move up the board"). For Black we mirror the board (python-chess `mirror()`
swaps colors and flips ranks), so the network and the policy always operate in a
White-to-move frame. That makes one network serve both colors.

Move encoding is the AlphaZero 8x8x73 = 4672 scheme, computed in the canonical
frame: index = from_square*73 + plane, where plane is one of
  - 0..55  : "queen" slides, 8 directions x 7 distances (queen promotions implied)
  - 56..63 : 8 knight moves
  - 64..72 : 9 underpromotions (3 pieces {N,B,R} x 3 file-deltas {-1,0,+1})
This is globally injective, so distinct legal moves never collide.
"""

from __future__ import annotations

import chess

ACTION_SIZE = 4672

# Queen slide directions (file_delta, rank_delta), a fixed order.
_QUEEN_DIRS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
_KNIGHT_DELTAS = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
_UNDER_PIECES = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def encode_move_canonical(from_sq: int, to_sq: int, promotion) -> int:
    """Encode a move already expressed in the canonical (White-up) frame."""
    ff, fr = from_sq & 7, from_sq >> 3
    tf, tr = to_sq & 7, to_sq >> 3
    df, dr = tf - ff, tr - fr

    # Underpromotion (knight/bishop/rook). Queen promotions fall through to the
    # queen-slide planes (promotion is implied by reaching the last rank).
    if promotion in _UNDER_PIECES:
        plane = 64 + _UNDER_PIECES[promotion] * 3 + (df + 1)
        return from_sq * 73 + plane

    # Knight move?
    if (abs(df), abs(dr)) in {(1, 2), (2, 1)}:
        plane = 56 + _KNIGHT_DELTAS.index((df, dr))
        return from_sq * 73 + plane

    # Queen slide: straight or diagonal.
    direction = (_sign(df), _sign(dr))
    distance = max(abs(df), abs(dr))
    plane = _QUEEN_DIRS.index(direction) * 7 + (distance - 1)
    return from_sq * 73 + plane


def encode_move(board: chess.Board, move: chess.Move) -> int:
    """Encode a real move on `board` into its canonical action index."""
    if board.turn == chess.WHITE:
        f, t = move.from_square, move.to_square
    else:
        # Mirror to the canonical White-up frame.
        f, t = chess.square_mirror(move.from_square), chess.square_mirror(move.to_square)
    return encode_move_canonical(f, t, move.promotion)


class ChessGame:
    action_size = ACTION_SIZE

    __slots__ = ("board", "_legal", "_indices")

    def __init__(self, board: chess.Board | None = None):
        self.board = board if board is not None else chess.Board()
        self._legal: list[chess.Move] | None = None
        self._indices: list[int] | None = None

    def _ensure_legal(self):
        if self._legal is None:
            self._legal = list(self.board.legal_moves)
            self._indices = [encode_move(self.board, m) for m in self._legal]

    @property
    def to_play(self) -> int:
        return 1 if self.board.turn == chess.WHITE else -1

    def legal_moves(self) -> list[int]:
        self._ensure_legal()
        return self._indices

    def apply(self, action: int) -> "ChessGame":
        self._ensure_legal()
        for j, idx in enumerate(self._indices):
            if idx == action:
                b = self.board.copy(stack=False)
                b.push(self._legal[j])
                return ChessGame(b)
        raise ValueError(f"action {action} not legal in this position")

    def is_terminal(self) -> bool:
        return self.board.is_game_over(claim_draw=True)

    def result(self) -> int:
        """Outcome from White (+1) perspective: +1 win, -1 loss, 0 draw."""
        outcome = self.board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            return 0
        return 1 if outcome.winner == chess.WHITE else -1

    @property
    def ply(self) -> int:
        return self.board.ply()

    def __str__(self) -> str:
        return str(self.board)
