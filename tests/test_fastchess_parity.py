"""Parity: FastChessGame (Rust-backed) must behave identically to ChessGame.

Requires the built `fastchess` extension. `chess_env_fast` auto-adds
`fastchess/pybuild` to sys.path, so no PYTHONPATH is needed for this test
(it also works with `PYTHONPATH=fastchess/pybuild`).

Run:  PYTHONPATH=. uv run python -m pytest tests/test_fastchess_parity.py -q
"""

from __future__ import annotations

import random

import numpy as np

from alphazero.chess_encode import encode_board
from alphazero.chess_env import ChessGame
from alphazero.chess_env_fast import FastChessGame


def _play_parallel(seed: int, max_ply: int, counters: dict):
    """Play one random game on both engines, comparing every position."""
    rng = random.Random(seed)
    cg = ChessGame()
    fg = FastChessGame()

    for _ in range(max_ply):
        counters["positions"] += 1

        # 1) legal action-index sets must be identical
        cg_legal = set(cg.legal_moves())
        fg_legal = set(fg.legal_moves())
        if cg_legal != fg_legal:
            counters["legal"] += 1
            counters["examples"].append(("legal", cg.board.fen(), cg_legal ^ fg_legal))

        # 2) to_play / ply
        if cg.to_play != fg.to_play:
            counters["to_play"] += 1
        if cg.ply != fg.ply:
            counters["ply"] += 1

        # 3) terminal + result
        cg_term, fg_term = cg.is_terminal(), fg.is_terminal()
        if cg_term != fg_term:
            counters["terminal"] += 1
            counters["examples"].append(("terminal", cg.board.fen(), (cg_term, fg_term)))
        if cg.result() != fg.result():
            counters["result"] += 1
            counters["examples"].append(("result", cg.board.fen(), (cg.result(), fg.result())))

        # 4) encode_board must produce identical planes (proves .board works)
        if not np.array_equal(encode_board(cg), encode_board(fg)):
            counters["encode"] += 1
            counters["examples"].append(("encode", cg.board.fen(), None))

        if cg_term:
            break

        # 5) apply the SAME action index on both, compare resulting position
        action = rng.choice(list(cg_legal))
        cg2 = cg.apply(action)
        fg2 = fg.apply(action)
        if cg2.board.fen() != fg2.board.fen():
            counters["apply"] += 1
            counters["examples"].append(("apply", cg.board.fen(), (cg2.board.fen(), fg2.board.fen())))
        cg, fg = cg2, fg2


def test_fastchess_parity_random_games():
    counters = {
        "positions": 0, "legal": 0, "to_play": 0, "ply": 0,
        "terminal": 0, "result": 0, "encode": 0, "apply": 0,
        "examples": [],
    }
    for seed in range(250):
        _play_parallel(seed, max_ply=120, counters=counters)

    assert counters["positions"] >= 3000, counters["positions"]
    mismatches = {k: counters[k] for k in
                  ("legal", "to_play", "ply", "terminal", "result", "encode", "apply")}
    assert all(v == 0 for v in mismatches.values()), (
        f"positions={counters['positions']} mismatches={mismatches} "
        f"examples={counters['examples'][:5]}"
    )


def test_search_action_parity():
    """A deterministic GumbelMCTS search must pick the SAME action on
    ChessGame and FastChessGame (this is what tactics_rates relies on)."""
    import chess
    import torch

    from alphazero.chess_net import ChessEvaluator, ChessNet
    from alphazero.gumbel import GumbelMCTS

    torch.manual_seed(0)
    net = ChessNet(channels=16, blocks=2)
    ev = ChessEvaluator(net, device="cpu")

    fens = [
        chess.STARTING_FEN,
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "4k3/8/4K3/8/8/8/8/7R w - - 0 1",  # mate-in-1 (Rh8#)
    ]
    for fen in fens:
        s_cg = ChessGame(chess.Board(fen))
        s_fg = FastChessGame(fen=fen)

        a_cg, _ = GumbelMCTS(ev, n_sims=32, max_considered=8,
                             rng=np.random.default_rng(123)).run(s_cg, add_noise=False)
        a_fg, _ = GumbelMCTS(ev, n_sims=32, max_considered=8,
                             rng=np.random.default_rng(123)).run(s_fg, add_noise=False)
        assert int(a_cg) == int(a_fg), (fen, int(a_cg), int(a_fg))
