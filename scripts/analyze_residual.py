"""Characterize the ~0.8% states where the trained AZ agent (100 sims) plays a
value-worsening move. Are they reachable under optimal play? Does more search
fix them?"""
import numpy as np
import torch
from collections import Counter

from alphazero.agents import AZAgent, RawNetAgent
from alphazero.evaluate import all_nonterminal_states
from alphazero.net import NetEvaluator, TicTacToeNet
from alphazero.solver import optimal_actions, solve

ck = torch.load("models/ttt_az.pt", weights_only=False)
net = TicTacToeNet(channels=ck["channels"])
net.load_state_dict(ck["state_dict"])
ev = NetEvaluator(net)

states = all_nonterminal_states()

def wrong_states(n_sims, seed=123):
    ag = AZAgent(ev, n_sims=n_sims, rng=np.random.default_rng(seed))
    bad = []
    for s in states:
        if ag.select(s) not in optimal_actions(s):
            bad.append(s)
    return bad

bad100 = wrong_states(100)
print(f"wrong states @100 sims: {len(bad100)} / {len(states)}  ({len(bad100)/len(states):.3%})")

# Categorize by solver value (side-to-move perspective) and piece count.
by_val = Counter(solve(s) for s in bad100)
by_pieces = Counter(sum(1 for v in s.board if v != 0) for s in bad100)
print("  by solver value (to-move):", dict(by_val), " (+1=winnable, 0=drawable; -1 impossible since all moves optimal)")
print("  by #pieces on board:", dict(sorted(by_pieces.items())))

# How severe: does the chosen move turn a win into <=draw, or a draw into a loss?
severity = Counter()
ag = AZAgent(ev, n_sims=100, rng=np.random.default_rng(123))
for s in bad100:
    v = solve(s)
    a = ag.select(s)
    v_after = -solve(s.apply(a))  # value to mover after the move
    severity[(v, v_after)] += 1
print("  (value_before -> value_after_move):", dict(severity))

# Does more search fix them?
for n in [200, 400, 800]:
    b = wrong_states(n)
    print(f"wrong states @{n} sims: {len(b)} ({len(b)/len(states):.3%})")

# Raw net (no search) for reference
raw = RawNetAgent(ev)
rb = sum(1 for s in states if raw.select(s) not in optimal_actions(s))
print(f"wrong states raw policy (0 sims): {rb} ({rb/len(states):.3%})")
