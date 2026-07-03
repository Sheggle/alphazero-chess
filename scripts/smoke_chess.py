import time, numpy as np, torch
torch.set_num_threads(1)
from alphazero.chess_env import ChessGame
from alphazero.chess_net import ChessNet, ChessEvaluator
from alphazero.gumbel import GumbelMCTS

net = ChessNet(channels=64, blocks=6)
print("params:", sum(p.numel() for p in net.parameters()))
ev = ChessEvaluator(net)

# Time a single self-play game (Gumbel, low sims, ply cap).
SIMS, MAXPLY, MC = 32, 60, 8
rng = np.random.default_rng(0)
g = ChessGame(); plies = 0; t0 = time.time(); nmoves=0
while not g.is_terminal() and plies < MAXPLY:
    a, pi = GumbelMCTS(ev, n_sims=SIMS, max_considered=MC, rng=rng).run(g, add_noise=True)
    assert a in g.legal_moves()
    g = g.apply(int(a)); plies += 1; nmoves+=1
dt = time.time() - t0
print(f"played {nmoves} plies in {dt:.1f}s -> {dt/nmoves*1000:.0f} ms/move, {SIMS} sims")
print(f"est ~{60/dt*nmoves/60:.1f} games/min single-thread at {MAXPLY}-ply cap" if dt>0 else "")
print("final terminal:", g.is_terminal(), "result:", g.result() if g.is_terminal() else "(capped)")
print("sample final position:\n", g)
