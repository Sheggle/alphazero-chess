"""Play-time repetition-aware move wrapper (shared by play_server.py and lichess_bot.py).

The deployed net (undiscounted ±1 value target, repetition-blind 18-plane encoding)
saturates its value near wins, so from a clearly-won position all winning moves look
equal and the search shuffles — often straight into a 3-fold draw. This wrapper fixes
that at the GAME-LOOP level (no search edit, no retraining):

  Only when the side to move is WINNING (root value > +0.5) AND the search's chosen
  move revisits a position the game has already been in, we do a 1-ply lookahead over
  the legal NON-repeating, non-drawing moves and instead play the one whose resulting
  position has the best net value (from the mover's POV). Otherwise the move is
  unchanged — so already-winning lines and non-winning positions behave exactly as before.

Verified on 6 Stockfish-confirmed wins: converted 3/6 -> 5/6, drew 3/6 -> 1/6.
"""
from __future__ import annotations
import chess

WIN_THRESH = 0.5


def position_key(board: chess.Board) -> str:
    """Position identity = FEN minus the two clock fields (placement, turn, castling, ep)."""
    return " ".join(board.fen().split()[:4])


def resolve_move(board: chess.Board, uci: str) -> chess.Move:
    """Resolve a (possibly promo-less) uci from the search to a concrete legal Move."""
    try:
        m = chess.Move.from_uci(uci)
    except ValueError:
        m = None
    if m in board.legal_moves:
        return m
    mq = chess.Move.from_uci(uci + "q")
    if mq in board.legal_moves:
        return mq
    for lm in board.legal_moves:
        if lm.uci()[:4] == uci[:4]:
            return lm
    raise ValueError(f"cannot resolve uci {uci!r} in {board.fen()}")


def _is_draw_child(child: chess.Board) -> bool:
    return (child.is_stalemate() or child.is_insufficient_material()
            or child.can_claim_draw())


def choose_move(board: chess.Board, chosen_uci: str, root_value, visited,
                value_batch, win_thresh: float = WIN_THRESH) -> str:
    """Return the uci to actually play.

    board       : position to move from (side to move = the mover).
    chosen_uci  : the search's chosen move.
    root_value  : search root value (mover's POV); may be None -> no override.
    visited     : set of position_key()s the game has already passed through.
    value_batch : callable(list[chess.Board]) -> sequence of net values (each board's
                  own side-to-move POV). Used only for the 1-ply lookahead.
    """
    mv = resolve_move(board, chosen_uci)
    if root_value is None or root_value <= win_thresh:
        return mv.uci()
    after = board.copy(); after.push(mv)
    if position_key(after) not in visited:
        return mv.uci()                       # not a repetition -> leave it
    # Repetition while winning: 1-ply lookahead over non-repeating, non-drawing moves.
    cand, childs = [], []
    for m in board.legal_moves:
        c = board.copy(); c.push(m)
        if position_key(c) in visited or _is_draw_child(c):
            continue
        cand.append(m); childs.append(c)
    if not cand:
        return mv.uci()                       # forced to repeat: keep the search move
    vals = value_batch(childs)                # child value is the OPPONENT's POV...
    best = min(range(len(cand)), key=lambda i: vals[i])  # ...so minimize it = maximize ours
    return cand[best].uci()
