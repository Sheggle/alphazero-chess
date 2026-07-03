"""Generate a FROZEN, objective chess tactics test suite.

We scan many *random-move* games (seeded, so the whole thing is deterministic)
and harvest two kinds of one-move tactics from the positions we pass through:

  * mate_in_1       — side to move has >=1 legal move that is checkmate.
                      Solution set = every legal move that mates.
  * hanging_capture — side to move can capture an undefended opponent piece worth
                      >=3 (N/B/R/Q): on the board *after* the capture, the
                      destination square is not attacked by the opponent, so the
                      piece is won clean. Solution set = every such capture.
                      We skip positions that also have a mate-in-1, so the
                      solution set is unambiguous.

Each entry is {"fen", "type", "solutions":[uci,...]}. Positions are de-duped by
FEN. The result is written to models/chess/tactics_suite.json.

Run:  PYTHONPATH=. uv run python scripts/gen_tactics.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import chess

OUT_PATH = Path("models/chess/tactics_suite.json")

# Material values; only captures of a piece worth >=3 count as "winning a piece".
_PIECE_VALUE = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 1000,
}


def mate_in_1_moves(board: chess.Board) -> list[str]:
    """UCI moves that deliver immediate checkmate."""
    out = []
    for mv in board.legal_moves:
        board.push(mv)
        if board.is_checkmate():
            out.append(mv.uci())
        board.pop()
    return out


def clean_capture_moves(board: chess.Board) -> list[str]:
    """UCI captures that win an opponent piece worth >=3 with no recapture.

    A move qualifies when:
      * it captures a non-pawn, non-king piece (value >= 3), and
      * en-passant is excluded (those capture a pawn anyway), and
      * after the capture the destination square is NOT attacked by the
        opponent (the piece cannot be recaptured) -> material won clean.
    """
    mover = board.turn
    opp = not mover
    out = []
    for mv in board.legal_moves:
        if not board.is_capture(mv):
            continue
        if board.is_en_passant(mv):
            continue  # captures a pawn (value 1)
        victim = board.piece_at(mv.to_square)
        if victim is None or _PIECE_VALUE[victim.piece_type] < 3:
            continue
        dest = mv.to_square
        board.push(mv)
        recapturable = board.is_attacked_by(opp, dest)
        board.pop()
        if not recapturable:
            out.append(mv.uci())
    return out


def generate(n_games: int, max_plies: int, target: int, seed: int):
    rng = random.Random(seed)
    suite: dict[str, dict] = {}  # fen -> entry (dedup by fen)
    n_mate = n_hang = 0

    for _ in range(n_games):
        if n_mate >= target and n_hang >= target:
            break
        board = chess.Board()
        for _ in range(max_plies):
            if board.is_game_over(claim_draw=True):
                break
            fen = board.fen()
            if fen not in suite:
                if n_mate < target:
                    mates = mate_in_1_moves(board)
                    if mates:
                        suite[fen] = {"fen": fen, "type": "mate_in_1", "solutions": mates}
                        n_mate += 1
                        # fall through to make a random move and continue
                if fen not in suite and n_hang < target:
                    # Only count as a clean-capture puzzle if there is no mate-in-1
                    # (so the solution set is the unambiguous "right" answer).
                    if not mate_in_1_moves(board):
                        caps = clean_capture_moves(board)
                        if caps:
                            suite[fen] = {
                                "fen": fen,
                                "type": "hanging_capture",
                                "solutions": caps,
                            }
                            n_hang += 1
            board.push(rng.choice(list(board.legal_moves)))

    return list(suite.values()), n_mate, n_hang


def sanity_check(entries: list[dict]) -> None:
    """Re-verify a few entries of each type from scratch."""
    by_type: dict[str, list[dict]] = {}
    for e in entries:
        by_type.setdefault(e["type"], []).append(e)

    for typ, items in by_type.items():
        print(f"  sanity-checking up to 5 {typ} entries...")
        for e in items[:5]:
            board = chess.Board(e["fen"])
            assert not board.is_game_over(claim_draw=True), "terminal position in suite"
            assert e["solutions"], "empty solution set"
            for uci in e["solutions"]:
                mv = chess.Move.from_uci(uci)
                assert mv in board.legal_moves, f"{uci} not legal in {e['fen']}"
                if typ == "mate_in_1":
                    board.push(mv)
                    ok = board.is_checkmate()
                    board.pop()
                    assert ok, f"{uci} is not mate in {e['fen']}"
                else:
                    assert board.is_capture(mv) and not board.is_en_passant(mv)
                    victim = board.piece_at(mv.to_square)
                    assert victim and _PIECE_VALUE[victim.piece_type] >= 3
                    opp = not board.turn
                    board.push(mv)
                    clean = not board.is_attacked_by(opp, mv.to_square)
                    board.pop()
                    assert clean, f"{uci} is recapturable in {e['fen']}"
    print("  sanity checks passed.")


def main() -> None:
    target = 50
    entries, n_mate, n_hang = generate(
        n_games=20000, max_plies=120, target=target, seed=20260627
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(entries, indent=2))

    print(f"Collected mate_in_1:       {n_mate}")
    print(f"Collected hanging_capture: {n_hang}")
    print(f"Total entries:             {len(entries)}")
    print(f"Wrote {OUT_PATH}")
    sanity_check(entries)


if __name__ == "__main__":
    main()
