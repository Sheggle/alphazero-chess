import random

from alphazero.agents import MCTSAgent, PerfectAgent, RandomAgent
from alphazero.arena import play_match


def test_mcts_100_outplays_random():
    """Step 3: 100-sim pure MCTS should crush random over a match."""
    mcts = MCTSAgent(n_sims=100, rng=random.Random(0))
    rand = RandomAgent(rng=random.Random(1000))
    res = play_match(mcts, rand, n_games=120)
    assert res.score > 0.85, res
    assert res.losses <= 2, res  # rare under-powered-opening blunders at most


def test_mcts_never_beats_perfect():
    """MCTS cannot beat game-theoretically optimal play."""
    mcts = MCTSAgent(n_sims=400, rng=random.Random(0))
    perfect = PerfectAgent(rng=random.Random(1))
    res = play_match(mcts, perfect, n_games=40)
    assert res.wins == 0, res


def test_perfect_never_loses_to_random():
    perfect = PerfectAgent(rng=random.Random(0))
    rand = RandomAgent(rng=random.Random(1))
    res = play_match(perfect, rand, n_games=100)
    assert res.losses == 0, res
