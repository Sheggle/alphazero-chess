"""Fast, objective chess-skill probe over a frozen tactics suite.

`tactics_rates` runs ONE greedy GumbelMCTS search per suite position (no Gumbel
noise -> deterministic) and checks whether the chosen move solves the tactic.
This is the primary skill metric for overnight hyperparameter tuning, so it is
deterministic given (evaluator, sims, max_considered, rng_seed).
"""

from __future__ import annotations

import json
from pathlib import Path

import chess
import numpy as np

from .chess_env import ChessGame, encode_move
from .gumbel import GumbelMCTS

DEFAULT_PATH = "models/chess/tactics_suite.json"


def load_suite(path: str = DEFAULT_PATH) -> list[dict]:
    return json.loads(Path(path).read_text())


def _action_to_uci(board: chess.Board, action: int) -> str | None:
    """Recover the UCI move whose canonical action index is `action`.

    Same trick ChessGame.apply uses: find the legal move that encodes to it.
    """
    for mv in board.legal_moves:
        if encode_move(board, mv) == action:
            return mv.uci()
    return None


def tactics_rates(
    evaluator,
    sims: int = 32,
    max_considered: int = 8,
    rng_seed: int = 0,
) -> dict:
    """Fraction of suite positions the net solves with one greedy search each.

    Returns {"mate_in_1": rate, "hanging_capture": rate, "overall": rate,
             "n": {"mate_in_1": k, "hanging_capture": k, "overall": k}}.
    """
    suite = load_suite()
    solved: dict[str, int] = {"mate_in_1": 0, "hanging_capture": 0}
    total: dict[str, int] = {"mate_in_1": 0, "hanging_capture": 0}

    for entry in suite:
        typ = entry["type"]
        total[typ] = total.get(typ, 0) + 1

        board = chess.Board(entry["fen"])
        state = ChessGame(board)
        search = GumbelMCTS(
            evaluator,
            n_sims=sims,
            max_considered=max_considered,
            rng=np.random.default_rng(rng_seed),
        )
        action, _ = search.run(state, add_noise=False)
        uci = _action_to_uci(board, action)
        if uci is not None and uci in set(entry["solutions"]):
            solved[typ] = solved.get(typ, 0) + 1

    def rate(s: int, t: int) -> float:
        return s / t if t else 0.0

    n_overall = sum(total.values())
    s_overall = sum(solved.values())
    return {
        "mate_in_1": rate(solved.get("mate_in_1", 0), total.get("mate_in_1", 0)),
        "hanging_capture": rate(
            solved.get("hanging_capture", 0), total.get("hanging_capture", 0)
        ),
        "overall": rate(s_overall, n_overall),
        "n": {
            "mate_in_1": total.get("mate_in_1", 0),
            "hanging_capture": total.get("hanging_capture", 0),
            "overall": n_overall,
        },
    }
