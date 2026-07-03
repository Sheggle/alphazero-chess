"""Aggregate baseline vs mix across training seeds. For each model eval the
primary 3-sim/w3 regime and the 32-sim/w8 stress regime."""
import random
import numpy as np, torch
from alphazero.gumbel import GumbelMCTS
from alphazero.net import NetEvaluator, TicTacToeNet
from alphazero.solver import optimal_actions
from alphazero.evaluate import all_nonterminal_states
from alphazero.arena import play_match
from alphazero.agents import PerfectAgent

S = all_nonterminal_states()
Ssamp = random.Random(0).sample(S, 1000)
SEEDS = [0, 1, 2]

def load(tag, s):
    ck = torch.load(f"models/ttt_mix_{tag}_s{s}.pt", weights_only=False)
    net = TicTacToeNet(channels=ck["channels"]); net.load_state_dict(ck["state_dict"])
    return NetEvaluator(net)

class Ag:
    def __init__(self, ev, n, mc): self.ev=ev; self.n=n; self.mc=mc
    def select(self, s):
        a,_ = GumbelMCTS(self.ev, n_sims=self.n, max_considered=self.mc,
                         rng=np.random.default_rng(0)).run(s, add_noise=False)
        return int(a)

def opt(ev,n,mc): 
    ag=Ag(ev,n,mc); return sum(ag.select(s) in optimal_actions(s) for s in Ssamp)/len(Ssamp)
def vp(ev,n,mc):  # total losses over 3 opponent seeds x 150 games = 450
    return sum(play_match(Ag(ev,n,mc),PerfectAgent(random.Random(p)),n_games=150).losses for p in range(3))

for regime,(n,mc) in {"3sim/w3 (primary)":(3,3), "32sim/w8 (stress)":(32,8)}.items():
    print(f"\n### regime {regime} ###")
    for tag in ["baseline","mix"]:
        rows=[]
        for s in SEEDS:
            ev=load(tag,s); rows.append((opt(ev,n,mc), vp(ev,n,mc)))
        opts=[r[0] for r in rows]; lps=[r[1] for r in rows]
        print(f"  {tag:8s}: opt={[f'{o:.3f}' for o in opts]} mean={np.mean(opts):.3f} | "
              f"vs_perfect losses/450 per seed={lps} mean={np.mean(lps):.1f}")
