import numpy as np

from alphazero.gumbel import GumbelMCTS
from alphazero.solver import optimal_actions, solve
from alphazero.tictactoe import TicTacToe


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


def test_gumbel_oracle_optimal_at_two_sims():
    """With a perfect evaluator, Gumbel (deterministic) plays optimally even at
    2 sims across a swath of positions."""
    orc = OracleEvaluator()
    rng = np.random.default_rng(0)
    from alphazero.evaluate import all_nonterminal_states
    bad = 0
    states = all_nonterminal_states()[:400]
    for s in states:
        a, _ = GumbelMCTS(orc, n_sims=2, rng=rng).run(s, add_noise=False)
        if a not in optimal_actions(s):
            bad += 1
    assert bad == 0, f"{bad}/{len(states)} suboptimal at 2 sims"


def test_gumbel_returns_valid_policy():
    orc = OracleEvaluator()
    rng = np.random.default_rng(1)
    a, pi = GumbelMCTS(orc, n_sims=4, rng=rng).run(TicTacToe(), add_noise=True)
    assert pi.shape == (9,)
    assert abs(pi.sum() - 1.0) < 1e-5
    # Illegal/occupied cells get zero mass (empty board: all legal, so check a
    # mid-game state).
    s = TicTacToe().apply(4)
    _, pi2 = GumbelMCTS(orc, n_sims=4, rng=rng).run(s, add_noise=True)
    assert pi2[4] == 0.0


def test_root_candidate_width_independent_of_sims():
    """Regression for the eval-width bug (OPTIMIZATION_LOG Change 4): the number
    of root candidates actually searched must stay <= max_considered regardless
    of the sim budget, so eval at more sims does not widen the candidate set
    (which would query the value head out-of-distribution)."""
    orc = OracleEvaluator()
    s = TicTacToe()  # 9 legal moves
    for sims in (3, 8, 32):
        m = GumbelMCTS(orc, n_sims=sims, max_considered=3,
                       rng=np.random.default_rng(0))
        touched = set()
        orig = m._simulate
        m._simulate = lambda root, a: (touched.add(a), orig(root, a))[1]
        m.run(s, add_noise=False)
        assert len(touched) <= 3, f"sims={sims}: searched {len(touched)} root actions"


def test_gumbel_respects_sim_budget():
    """Total simulations spent at the root should not exceed n_sims."""
    orc = OracleEvaluator()
    rng = np.random.default_rng(2)
    s = TicTacToe()
    m = GumbelMCTS(orc, n_sims=5, rng=rng)
    # Patch _simulate to count calls.
    calls = {"n": 0}
    orig = m._simulate
    m._simulate = lambda root, a: (calls.__setitem__("n", calls["n"] + 1), orig(root, a))[1]
    m.run(s, add_noise=False)
    assert calls["n"] <= 5, calls["n"]
