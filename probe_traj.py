import sys
from pathlib import Path
import torch, chess
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "fastchess" / "pybuild"))
from alphazero.chess_env import ChessGame
from alphazero.chess_net import ChessNet, ChessEvaluator

ck = torch.load(ROOT / "models/chess_gpu/playnet.pt", map_location="cpu")
net = ChessNet(ck["channels"], ck["blocks"]); net.load_state_dict(ck["state_dict"]); net.eval()
dev = "mps" if torch.backends.mps.is_available() else "cpu"
ev = ChessEvaluator(net, device=dev)

moves = "e4 d5 exd5 e6 dxe6 Bxe6 Nf3 c5 Nc3 Nc6 Bb5 a6 Ba4 b5 Bb3 c4 O-O b4 Ne4 Nf6 Nxf6+ gxf6 Ba4 Qd6".split()
b = chess.Board()


def blackval(board):
    _, v = ev.predict(ChessGame(board.copy()))
    return v if board.turn == chess.BLACK else -v  # always Black's perspective


print(f"net frames {ck.get('frames',0)/1e6:.0f}M   (value from BLACK's perspective, +=Black better)\n")
print(f"  start                         {blackval(b):+.3f}")
for i, mv in enumerate(moves):
    san = mv
    b.push_san(mv)
    n = i // 2 + 1
    who = "W" if i % 2 == 0 else "B"
    tag = ""
    if san == "c4": tag = "  <- c4 traps Bb3 (Black can win it w/ cxb3)"
    if san == "b4": tag = "  <- RELEASES the pressure (b4 instead of cxb3)"
    if san == "Ba4" and i > 20: tag = "  <- your bishop ESCAPES"
    print(f"  {n:2d}{who} {san:7s}                  {blackval(b):+.3f}{tag}")
# also: the bishop-winning alternative at move 9
b2 = chess.Board()
for mv in moves[:17]:  # up to and including 9.O-O
    b2.push_san(mv)
b2.push_san("cxb3"); b2.push_san("axb3")
print(f"\n  ALT 9...cxb3 10.axb3 (took the bishop): {blackval(b2):+.3f}  (Black, after winning B-for-P)")
