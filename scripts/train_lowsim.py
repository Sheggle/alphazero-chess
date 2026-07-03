"""Train the recommended robust low-sim (3-sim) agent, save it, and run the
final evaluation that matters for play strength: at 3 sims, does it ever LOSE?
(opt-rate measures move-optimality; this measures outcomes.)"""
import random
from pathlib import Path

import numpy as np
import torch

from alphazero.agents import GumbelAgent, PerfectAgent, RandomAgent, RawNetAgent
from alphazero.arena import play_game, play_match
from alphazero.evaluate import all_nonterminal_states, optimal_move_rate
from alphazero.net import NetEvaluator, TicTacToeNet
from alphazero.train import robust_lowsim_config, train


def main():
    cfg = robust_lowsim_config(iterations=50)
    net, evaluator, cfg = train(cfg)

    Path("models").mkdir(exist_ok=True)
    ckpt = Path("models/ttt_gumbel_3sim.pt")
    torch.save({"state_dict": net.state_dict(), "channels": cfg.channels}, ckpt)
    print(f"\nsaved -> {ckpt}")

    print("\n=== FINAL EVAL @ 3 sims ===")
    states = all_nonterminal_states()
    g = lambda s=0: GumbelAgent(evaluator, n_sims=3, c_visit=cfg.c_visit,
                                c_scale=cfg.c_scale, rng=np.random.default_rng(s))
    print(f"opt(mcts) 3 sims, ALL {len(states)} states: "
          f"{optimal_move_rate(g(7), states):.4f}")
    print(f"opt(raw)  ALL states:                      "
          f"{optimal_move_rate(RawNetAgent(evaluator), states):.4f}")

    vr = play_match(g(1), RandomAgent(random.Random(1)), n_games=400)
    print(f"vs random (400): {vr}   losses={vr.losses} (want 0)")
    vp = play_match(g(2), PerfectAgent(random.Random(2)), n_games=200)
    print(f"vs perfect (200): {vp}   losses={vp.losses}")
    draws = sum(play_game(g(s), g(s + 500)) == 0 for s in range(50))
    print(f"self-play draws: {draws}/50")

    # Width-matched deep eval (the Change-4 fix): the SAME 3-sim-trained net,
    # searched deeper at the training width, should play perfectly.
    print("\n=== deep eval at training width (max_considered=3) ===")
    gd = lambda n, s: GumbelAgent(evaluator, n_sims=n, max_considered=3,
                                  c_visit=cfg.c_visit, c_scale=cfg.c_scale,
                                  rng=np.random.default_rng(s))
    for n in [3, 16, 32]:
        opt = optimal_move_rate(gd(n, 7), states)
        lp = [play_match(gd(n, 2), PerfectAgent(random.Random(ps)), n_games=200).losses
              for ps in range(4)]
        print(f"  {n:2d} sims: opt={opt:.4f}  vs_perfect losses/200 (4 seeds)={lp}")


if __name__ == "__main__":
    main()
