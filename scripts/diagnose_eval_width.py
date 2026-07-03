"""Confirm the fix: hold candidate width = training width (max_considered=3) and
verify eval sims becomes a pure depth knob (monotone-good, no spike)."""
import random, numpy as np, torch
from alphazero.gumbel import GumbelMCTS
from alphazero.net import NetEvaluator, TicTacToeNet
from alphazero.solver import optimal_actions
from alphazero.evaluate import all_nonterminal_states
from alphazero.arena import play_match
from alphazero.agents import PerfectAgent, RandomAgent

ck=torch.load("models/ttt_gumbel_3sim.pt",weights_only=False)
net=TicTacToeNet(channels=ck["channels"]); net.load_state_dict(ck["state_dict"])
ev=NetEvaluator(net); S=all_nonterminal_states()

class Ag:
    def __init__(self,n,mc): self.n=n; self.mc=mc
    def select(self,s):
        a,_=GumbelMCTS(ev,n_sims=self.n,max_considered=self.mc,rng=np.random.default_rng(0)).run(s,add_noise=False)
        return int(a)

for mc in [8, 3]:
    print(f"\n=== max_considered={mc} (8=buggy default, 3=fix) ===")
    for n in [3,8,16,32,64]:
        ag=Ag(n,mc)
        opt=sum(ag.select(s) in optimal_actions(s) for s in S)/len(S)
        lp=[play_match(Ag(n,mc),PerfectAgent(random.Random(ps)),n_games=200).losses for ps in range(6)]
        lr=play_match(Ag(n,mc),RandomAgent(random.Random(1)),n_games=400).losses
        print(f"  {n:2d} sims: opt={opt:.4f}  vs_perfect mean={np.mean(lp):4.1f} {lp}  vs_random={lr}/400")
