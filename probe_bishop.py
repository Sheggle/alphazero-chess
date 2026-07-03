import sys
from pathlib import Path
import numpy as np, torch, chess
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "fastchess" / "pybuild"))
from alphazero.chess_env import ChessGame, encode_move
from alphazero.chess_net import ChessNet, ChessEvaluator
from alphazero.gumbel import GumbelMCTS

# position after 9.O-O, Black to move; c4-pawn attacks the Bb3 (defended by a2). ...cxb3 wins B for P.
FEN = "r2qkbnr/5ppp/p1n1b3/1p6/2p5/1BN2N2/PPPP1PPP/R1BQ1RK1 b kq - 0 9"
ck = torch.load(ROOT / "models/chess_gpu/playnet.pt", map_location="cpu")
net = ChessNet(ck["channels"], ck["blocks"]); net.load_state_dict(ck["state_dict"]); net.eval()
dev = "mps" if torch.backends.mps.is_available() else "cpu"
ev = ChessEvaluator(net, device=dev)
b = chess.Board(FEN); g = ChessGame(b)
probs, val = ev.predict(g)
legal = list(b.legal_moves)
ranked = sorted(legal, key=lambda m: -probs[encode_move(b, m)])
print(f"net value (Black to move, Black's perspective): {val:+.3f}  (frames {ck.get('frames',0)/1e6:.0f}M)")
print("top policy priors:")
for i, m in enumerate(ranked[:14]):
    tag = "   <== cxb3 wins bishop-for-pawn" if m.uci() == "c4b3" else ""
    print(f"  {i+1:2d}. {b.san(m):7s} {probs[encode_move(b,m)]*100:5.1f}%{tag}")
cx = chess.Move.from_uci("c4b3")
rk = [m.uci() for m in ranked].index("c4b3") + 1
print(f"\ncxb3 policy rank: {rk}/{len(legal)}  (Gumbel root only searches the top max_considered=16)")
for sims in (128, 400):
    a, _ = GumbelMCTS(ev, n_sims=sims, max_considered=16, c_scale=0.3,
                      rng=np.random.default_rng(0)).run(g, add_noise=False)
    pick = next(m for m in legal if encode_move(b, m) == a)
    print(f"search {sims} sims picks: {b.san(pick)}")
g2 = g.apply(encode_move(b, cx)); _, v2 = ev.predict(g2)
print(f"\nvalue AFTER ...cxb3 (White to move, White's persp): {v2:+.3f}  -> Black persp {-v2:+.3f}")
print(f"value of current pos (Black persp): {val:+.3f}   (if cxb3 really wins, -v2 should be >> val)")
