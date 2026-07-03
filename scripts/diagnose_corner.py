"""Is the corner-opening blunder a bug or just under-powered search?
Take the exact position (X at corner 2, O to move; only optimal reply is center
4) and see how MCTS's choice + visit distribution change with more sims."""
import random
from collections import Counter
from alphazero.mcts import MCTS
from alphazero.tictactoe import TicTacToe

s = TicTacToe().apply(2)  # X plays corner 2 -> O to move. Optimal: center (4).
print("Position (O to move), optimal reply = center (4):")
print(s, "\n")

for n_sims in [50, 100, 200, 400, 800, 1600]:
    picks = Counter()
    for seed in range(40):
        m = MCTS(n_sims=n_sims, rng=random.Random(seed))
        picks[m.search(s)] += 1
    center_rate = picks[4] / 40
    print(f"n_sims={n_sims:5d}: center(4) picked {picks[4]:2d}/40 ({center_rate:.0%})  full={dict(picks)}")
