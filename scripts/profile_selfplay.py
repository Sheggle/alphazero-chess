import cProfile, pstats, io, time
import numpy as np, torch
torch.set_num_threads(1)
from alphazero.chess_env import ChessGame
from alphazero.chess_net import ChessNet, ChessEvaluator
from alphazero.gumbel import GumbelMCTS

net = ChessNet(channels=64, blocks=6); ev = ChessEvaluator(net)
rng = np.random.default_rng(0)

def play():
    g = ChessGame(); p=0
    while not g.is_terminal() and p < 40:
        a,_ = GumbelMCTS(ev, n_sims=32, max_considered=8, rng=rng).run(g, add_noise=True)
        g = g.apply(int(a)); p+=1

# time net.forward batch-1 vs batch-64 first
import torch as T
x1 = T.randn(1,18,8,8); x64=T.randn(64,18,8,8)
net.eval()
with T.no_grad():
    for _ in range(3): net(x1); net(x64)
    t=time.time()
    for _ in range(50): net(x1)
    t1=(time.time()-t)/50*1000
    t=time.time()
    for _ in range(20): net(x64)
    t64=(time.time()-t)/20*1000
print(f"net forward: batch1={t1:.2f}ms  batch64={t64:.2f}ms  per-pos-in-batch={t64/64:.3f}ms  speedup={t1/(t64/64):.1f}x")

pr = cProfile.Profile(); pr.enable(); play(); pr.disable()
s = io.StringIO(); ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
ps.print_stats(20)
print(s.getvalue())
