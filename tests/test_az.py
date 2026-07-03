import numpy as np

from alphazero.az_mcts import AZMCTS, policy_from_visits
from alphazero.encoder import encode, symmetries
from alphazero.net import NetEvaluator, TicTacToeNet
from alphazero.selfplay import play_selfplay_game
from alphazero.solver import optimal_actions, solve
from alphazero.tictactoe import TicTacToe


def test_encode_perspective():
    # After X plays center, it's O to move; plane 0 is the side-to-move (O).
    s = TicTacToe().apply(4)
    planes = encode(s)
    assert planes.shape == (2, 3, 3)
    assert planes[1, 1, 1] == 1.0  # opponent (X) occupies center
    assert planes[0].sum() == 0.0  # O has nothing yet


def test_symmetries_count_and_validity():
    s = TicTacToe().apply(0)  # one corner -> 8 distinct symmetries... but corner
    pi = np.zeros(9, dtype=np.float32)
    pi[1] = 1.0
    planes = encode(s)
    syms = list(symmetries(planes, pi))
    # Each policy stays a valid one-hot distribution summing to 1.
    for p, q in syms:
        assert p.shape == (2, 3, 3)
        assert abs(q.sum() - 1.0) < 1e-6
    # Identity must be present.
    assert any(np.array_equal(p, planes) and np.array_equal(q, pi) for p, q in syms)


def test_net_output_shapes():
    net = TicTacToeNet(channels=8)
    ev = NetEvaluator(net)
    probs, value = ev.predict(TicTacToe())
    assert probs.shape == (9,)
    assert abs(probs.sum() - 1.0) < 1e-5
    assert -1.0 <= value <= 1.0


def test_net_masks_illegal_moves():
    net = TicTacToeNet(channels=8)
    ev = NetEvaluator(net)
    s = TicTacToe().apply(4)  # cell 4 occupied
    probs, _ = ev.predict(s)
    assert probs[4] == 0.0


class OracleEvaluator:
    def predict(self, state):
        if state.is_terminal():
            return np.ones(9) / 9, float(state.result() * state.to_play)
        opt = set(optimal_actions(state))
        probs = np.zeros(9)
        for a in state.legal_moves():
            probs[a] = 1.0 if a in opt else 0.1
        probs /= probs.sum()
        return probs, float(solve(state))


def test_azmcts_with_oracle_plays_optimally():
    oracle = OracleEvaluator()
    # corner opening -> center is the only optimal reply
    s = TicTacToe().apply(2)
    mcts = AZMCTS(oracle, n_sims=50, rng=np.random.default_rng(0))
    visits = mcts.run(s)
    pi = policy_from_visits(visits, 9, temperature=0.0)
    assert int(pi.argmax()) in optimal_actions(s)


def test_policy_from_visits_temperatures():
    visits = {0: 10, 1: 30, 2: 60}
    greedy = policy_from_visits(visits, 9, temperature=0.0)
    assert greedy.argmax() == 2 and greedy[2] == 1.0
    prop = policy_from_visits(visits, 9, temperature=1.0)
    assert abs(prop.sum() - 1.0) < 1e-6
    assert abs(prop[2] - 0.6) < 1e-6


def test_selfplay_targets_consistent():
    net = TicTacToeNet(channels=8)
    ev = NetEvaluator(net)
    samples = play_selfplay_game(ev, n_sims=10, rng=np.random.default_rng(0))
    assert len(samples) >= 5
    for s in samples:
        assert abs(s.pi.sum() - 1.0) < 1e-5
        assert s.z in (-1.0, 0.0, 1.0)
