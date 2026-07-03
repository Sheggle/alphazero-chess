"""Self-consistency + smoke tests for the frozen tactics suite and probe."""

from __future__ import annotations

import chess
import numpy as np
import pytest

from alphazero.chess_env import ACTION_SIZE
from alphazero.chess_net import ChessEvaluator, ChessNet
from alphazero.chess_tactics import load_suite, tactics_rates

_PIECE_VALUE = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 1000,
}


class RandomEvaluator:
    """Uniform policy over legal moves, value 0 — a no-skill baseline."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def predict(self, state):
        legal = state.legal_moves()
        probs = np.zeros(ACTION_SIZE, dtype=np.float32)
        if legal:
            probs[legal] = 1.0 / len(legal)
        return probs, 0.0


def test_suite_nonempty_with_both_types():
    suite = load_suite()
    assert suite, "suite is empty"
    types = {e["type"] for e in suite}
    assert "mate_in_1" in types
    assert "hanging_capture" in types


def test_suite_is_self_consistent():
    """Every listed solution genuinely satisfies its tactic when replayed."""
    suite = load_suite()
    for e in suite:
        board = chess.Board(e["fen"])
        assert not board.is_game_over(claim_draw=True), f"terminal: {e['fen']}"
        assert e["solutions"], f"no solutions: {e['fen']}"
        legal = set(board.legal_moves)
        for uci in e["solutions"]:
            mv = chess.Move.from_uci(uci)
            assert mv in legal, f"{uci} illegal in {e['fen']}"
            if e["type"] == "mate_in_1":
                board.push(mv)
                ok = board.is_checkmate()
                board.pop()
                assert ok, f"{uci} not mate in {e['fen']}"
            elif e["type"] == "hanging_capture":
                assert board.is_capture(mv) and not board.is_en_passant(mv)
                victim = board.piece_at(mv.to_square)
                assert victim is not None
                assert _PIECE_VALUE[victim.piece_type] >= 3
                opp = not board.turn
                board.push(mv)
                clean = not board.is_attacked_by(opp, mv.to_square)
                board.pop()
                assert clean, f"{uci} recapturable in {e['fen']}"
            else:
                pytest.fail(f"unknown type {e['type']}")


def test_random_agent_scores_lowish():
    rates = tactics_rates(RandomEvaluator(seed=0), sims=16, max_considered=8)
    # A no-skill agent should rarely stumble onto the exact tactic.
    assert rates["overall"] < 0.35, rates


def test_fresh_net_runs_without_error():
    net = ChessNet()
    ev = ChessEvaluator(net)
    rates = tactics_rates(ev, sims=16, max_considered=8)
    assert set(rates) == {"mate_in_1", "hanging_capture", "overall", "n"}
    assert rates["n"]["overall"] == rates["n"]["mate_in_1"] + rates["n"]["hanging_capture"]
    for k in ("mate_in_1", "hanging_capture", "overall"):
        assert 0.0 <= rates[k] <= 1.0
