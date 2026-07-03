import random

from alphazero.mcts import MCTS
from alphazero.solver import optimal_actions
from alphazero.tictactoe import TicTacToe


def test_takes_immediate_win():
    # X at 0,1 -> X to move should take the win at 2.
    s = TicTacToe()
    for a in [0, 3, 1, 4]:  # X0 O3 X1 O4 -> X to move, win at 2
        s = s.apply(a)
    assert s.to_play == 1
    mcts = MCTS(n_sims=200, rng=random.Random(0))
    assert mcts.search(s) == 2


def test_blocks_immediate_threat():
    # X threatens at 2; O (to move) must block there.
    s = TicTacToe()
    for a in [0, 4, 1]:  # X0 O4 X1 -> O to move, block at 2
        s = s.apply(a)
    assert s.to_play == -1
    mcts = MCTS(n_sims=400, rng=random.Random(0))
    assert mcts.search(s) == 2


def test_picks_only_optimal_in_fork_setup():
    # A position with a unique best move per the solver; MCTS with enough
    # sims should land on one of the optimal moves.
    s = TicTacToe()
    for a in [4, 0]:  # X center, O corner
        s = s.apply(a)
    mcts = MCTS(n_sims=600, rng=random.Random(1))
    move = mcts.search(s)
    assert move in optimal_actions(s)


def test_visit_distribution_concentrates_on_win():
    s = TicTacToe()
    for a in [0, 3, 1, 4]:
        s = s.apply(a)
    mcts = MCTS(n_sims=300, rng=random.Random(2))
    visits = mcts.action_visits(s)
    # The winning move (2) should be the most visited by a clear margin.
    assert max(visits, key=visits.get) == 2
