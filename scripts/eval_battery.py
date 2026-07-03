"""Robustness battery for Step 3."""
import random
from alphazero.agents import MCTSAgent, RandomAgent, PerfectAgent
from alphazero.arena import play_match

print("MCTS(100) vs Random, 200 games per seed:")
agg = []
for seed in range(5):
    mcts = MCTSAgent(n_sims=100, rng=random.Random(seed))
    rand = RandomAgent(rng=random.Random(1000 + seed))
    r = play_match(mcts, rand, n_games=200)
    agg.append(r.score)
    print(f"  seed {seed}: {r}")
print(f"  mean score over seeds: {sum(agg)/len(agg):.3f}\n")

print("MCTS(800) vs Random (should ~never lose):")
mcts = MCTSAgent(n_sims=800, rng=random.Random(0))
rand = RandomAgent(rng=random.Random(99))
print("  ", play_match(mcts, rand, n_games=200))

print("\nPerfect vs Random (ceiling):")
print("  ", play_match(PerfectAgent(random.Random(0)), RandomAgent(random.Random(1)), n_games=200))

print("\nMCTS(800) vs Perfect (should be all draws — both ~optimal):")
print("  ", play_match(MCTSAgent(n_sims=800, rng=random.Random(0)), PerfectAgent(random.Random(1)), n_games=100))
