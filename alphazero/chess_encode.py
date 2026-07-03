"""Encode a chess position into network input planes (canonical, White-up).

18 planes of 8x8:
  0..5   : side-to-move's pieces  (P,N,B,R,Q,K)
  6..11  : opponent's pieces      (P,N,B,R,Q,K)
  12..15 : castling rights (my K, my Q, opp K, opp Q)
  16     : en-passant target square
  17     : halfmove (fifty-move) clock, scaled to [0,1]
"""

from __future__ import annotations

import chess
import numpy as np

INPUT_PLANES = 18
_PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def canonical_board(board: chess.Board) -> chess.Board:
    """Board as seen by the side to move (White to move always)."""
    return board if board.turn == chess.WHITE else board.mirror()


def encode_board(game) -> np.ndarray:
    board = canonical_board(game.board)  # White to move in this frame
    planes = np.zeros((INPUT_PLANES, 8, 8), dtype=np.float32)

    for sq, piece in board.piece_map().items():
        r, f = sq >> 3, sq & 7
        base = 0 if piece.color == chess.WHITE else 6
        planes[base + _PIECE_ORDER.index(piece.piece_type), r, f] = 1.0

    if board.has_kingside_castling_rights(chess.WHITE):
        planes[12] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[13] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[14] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[15] = 1.0

    if board.ep_square is not None:
        planes[16, board.ep_square >> 3, board.ep_square & 7] = 1.0

    planes[17] = min(board.halfmove_clock, 100) / 100.0
    return planes


def encode_state(game) -> np.ndarray:
    """Encode a game state to (18,8,8) float32 planes.

    Uses the game's own fast Rust `encode()` (FastChessGame) when available —
    which fills the buffer directly from the board with no python-chess walk —
    and otherwise falls back to the reference `encode_board`. The two are
    bit-for-bit identical, so the search/training algorithm is unchanged.
    """
    enc = getattr(game, "encode", None)
    return enc() if enc is not None else encode_board(game)
