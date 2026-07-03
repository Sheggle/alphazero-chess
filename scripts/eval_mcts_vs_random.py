"""Step 3 verification: does 100-sim pure MCTS outplay a random agent?

Plays a match (alternating sides) and prints the tally. We expect MCTS to win
the large majority of games and essentially never lose to random play.
"""
import random
import sys

from alphazero.agents import MCTSAgent, RandomAgent
from alphazero.arena import play_match


def main(n_games: int = 200, n_sims: int = 100, seed: int = 0):
    rng_mcts = random.Random(seed)
    rng_rand = random.Random(seed + 1)
    mcts = MCTSAgent(n_sims=n_sims, rng=rng_mcts)
    rand = RandomAgent(rng=rng_rand)
    res = play_match(mcts, rand, n_games=n_games)
    print(f"MCTS({n_sims} sims) vs Random over {n_games} games:")
    print("  ", res)
    print(f"   losses to random: {res.losses}  (want 0)")
    return res


if __name__ == "__main__":
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    n_sims = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    main(n_games, n_sims)
