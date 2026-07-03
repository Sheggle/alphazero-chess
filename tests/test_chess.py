import random

import chess
import numpy as np

from alphazero.chess_encode import INPUT_PLANES, encode_board
from alphazero.chess_env import ACTION_SIZE, ChessGame, encode_move


def test_action_size_and_basic_interface():
    g = ChessGame()
    assert g.action_size == ACTION_SIZE
    assert g.to_play == 1
    assert len(g.legal_moves()) == 20  # 20 opening moves
    assert not g.is_terminal()


def test_encode_move_injective_over_random_games():
    """For every position in many random games, all legal moves must encode to
    distinct indices in [0, 4672) (else MCTS would conflate moves)."""
    rng = random.Random(0)
    for _ in range(40):
        board = chess.Board()
        for _ in range(60):
            if board.is_game_over(claim_draw=True):
                break
            g = ChessGame(board)
            idxs = g.legal_moves()
            assert len(idxs) == len(set(idxs)), "collision among legal moves"
            assert all(0 <= i < ACTION_SIZE for i in idxs)
            board.push(rng.choice(list(board.legal_moves)))


def test_apply_roundtrip_matches_pushed_move():
    """apply(index) must reach the same position as pushing the intended move."""
    rng = random.Random(1)
    for _ in range(40):
        board = chess.Board()
        for _ in range(60):
            if board.is_game_over(claim_draw=True):
                break
            g = ChessGame(board)
            move = rng.choice(list(board.legal_moves))
            idx = encode_move(board, move)
            child = g.apply(idx)
            expected = board.copy(stack=False)
            expected.push(move)
            assert child.board.fen() == expected.fen()
            board.push(move)


def test_underpromotion_encodes_distinctly():
    # White pawn on a7 with promotion options; all four promotions must differ.
    board = chess.Board("8/P7/8/8/8/8/8/k6K w - - 0 1")
    g = ChessGame(board)
    promos = [m for m in board.legal_moves if m.promotion is not None]
    idxs = [encode_move(board, m) for m in promos]
    assert len(promos) == 4  # Q, R, B, N
    assert len(set(idxs)) == 4


def test_black_to_move_is_canonicalized():
    # After 1. e4, Black to move. Encoding/round-trip must still work (mirror).
    board = chess.Board()
    board.push_san("e4")
    g = ChessGame(board)
    assert g.to_play == -1
    move = chess.Move.from_uci("e7e5")
    idx = encode_move(board, move)
    child = g.apply(idx)
    expected = board.copy(stack=False)
    expected.push(move)
    assert child.board.fen() == expected.fen()


def test_result_perspective():
    # Fool's mate: White is checkmated -> result -1 (from White's perspective).
    board = chess.Board()
    for mv in ["f3", "e5", "g4", "Qh4#"]:
        board.push_san(mv)
    g = ChessGame(board)
    assert g.is_terminal()
    assert g.result() == -1


def test_encoder_shape_and_perspective():
    g = ChessGame()
    planes = encode_board(g)
    assert planes.shape == (INPUT_PLANES, 8, 8)
    # Start position: side-to-move pawns on rank 2 (row index 1).
    assert planes[0, 1, :].sum() == 8
    # Opponent pawns on rank 7 (row index 6).
    assert planes[6, 6, :].sum() == 8

    # After 1.e4 (Black to move), canonical frame flips: "my" pawns again on row1.
    board = chess.Board(); board.push_san("e4")
    planes_b = encode_board(ChessGame(board))
    assert planes_b[0, 1, :].sum() == 8  # mover (Black) pawns canonicalized to row1
