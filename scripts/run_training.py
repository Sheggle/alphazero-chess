"""Full AlphaZero training run for tic-tac-toe + thorough final evaluation.

Saves the trained net to models/ttt_az.pt and prints a final report:
optimal-move rate over ALL reachable states, match vs random, vs perfect, and a
self-play-draw check.
"""
import random
from pathlib import Path

import numpy as np
import torch

from alphazero.agents import AZAgent, PerfectAgent, RandomAgent, RawNetAgent
from alphazero.arena import play_game, play_match
from alphazero.evaluate import all_nonterminal_states, optimal_move_rate
from alphazero.train import Config, train


def main():
    cfg = Config(
        iterations=30,
        games_per_iter=40,
        n_sims=100,
        train_steps=150,
        batch_size=256,
        eval_random_games=200,
        eval_mcts_sample=700,
        seed=0,
    )
    net, evaluator, cfg = train(cfg)

    Path("models").mkdir(exist_ok=True)
    ckpt = Path("models/ttt_az.pt")
    torch.save({"state_dict": net.state_dict(), "channels": cfg.channels}, ckpt)
    print(f"\nsaved checkpoint -> {ckpt}")

    print("\n=== FINAL EVALUATION ===")
    states = all_nonterminal_states()
    az = AZAgent(evaluator, n_sims=cfg.n_sims, rng=np.random.default_rng(123))
    raw = RawNetAgent(evaluator)

    print(f"optimal-move rate (MCTS, ALL {len(states)} states): "
          f"{optimal_move_rate(az, states):.4f}")
    print(f"optimal-move rate (raw net, ALL states):           "
          f"{optimal_move_rate(raw, states):.4f}")

    vs_rand = play_match(az, RandomAgent(random.Random(1)), n_games=400)
    print(f"vs random (400 games):  {vs_rand}")
    vs_perf = play_match(az, PerfectAgent(random.Random(2)), n_games=200)
    print(f"vs perfect (200 games): {vs_perf}  (want 0 wins, 0 losses -> all draws)")

    # AZ vs AZ self-play should be a draw if it plays perfectly.
    draws = 0
    for s in range(50):
        r = play_game(
            AZAgent(evaluator, n_sims=cfg.n_sims, rng=np.random.default_rng(s)),
            AZAgent(evaluator, n_sims=cfg.n_sims, rng=np.random.default_rng(s + 1000)),
        )
        draws += (r == 0)
    print(f"AZ vs AZ self-play: {draws}/50 draws (want 50)")


if __name__ == "__main__":
    main()
