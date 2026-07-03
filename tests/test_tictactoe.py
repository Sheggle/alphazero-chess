from alphazero.solver import optimal_actions, solve
from alphazero.tictactoe import TicTacToe


def test_initial_state():
    s = TicTacToe()
    assert s.to_play == 1
    assert len(s.legal_moves()) == 9
    assert not s.is_terminal()


def test_apply_alternates_player():
    s = TicTacToe().apply(4)
    assert s.board[4] == 1
    assert s.to_play == -1
    s2 = s.apply(0)
    assert s2.board[0] == -1
    assert s2.to_play == 1


def test_illegal_move_raises():
    s = TicTacToe().apply(4)
    try:
        s.apply(4)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on occupied cell")


def test_row_win():
    # X plays 0,1,2 ; O plays 3,4 in between.
    s = TicTacToe()
    for a in [0, 3, 1, 4, 2]:
        s = s.apply(a)
    assert s.is_terminal()
    assert s.winner() == 1
    assert s.result() == 1


def test_draw():
    # A known drawn fill: X O X / X X O / O X O
    s = TicTacToe()
    for a in [0, 1, 2, 4, 3, 5, 7, 6, 8]:
        s = s.apply(a)
    assert s.is_terminal()
    assert s.winner() == 0
    assert s.result() == 0


def test_perfect_play_is_a_draw():
    # The whole point of TTT: optimal value of the empty board is 0.
    assert solve(TicTacToe()) == 0


def test_center_and_corners_are_optimal_openings():
    # From the empty board every optimal opening draws; classic theory says
    # corner/center/edge all hold the draw, so all 9 are "optimal" (value 0).
    opt = optimal_actions(TicTacToe())
    assert set(opt) == set(range(9))


def test_must_block_immediate_threat():
    # X at 0 and 1 threatens to win at 2; O (to move) must play 2.
    s = TicTacToe()
    for a in [0, 4, 1]:  # X0, O4, X1  -> O to move, must block at 2
        s = s.apply(a)
    assert s.to_play == -1
    assert optimal_actions(s) == [2]
