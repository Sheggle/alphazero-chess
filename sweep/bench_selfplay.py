"""Self-play throughput bench — fastchess.run_selfplay at the production config
(128ch/10b, sims=64, n_games=1024) on a CLEAN GPU. Reports pos/s (median over runs).
Used to compare throughput optimizations on separate branches (compile / refill / pipeline).
--compile torch.compiles the forward (opt #1). Same eval_fn/config across all variants.

  PYTHONPATH=.:fastchess/pybuild python sweep/bench_selfplay.py [--compile] [--runs N]
"""
import sys, time, argparse, statistics
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "fastchess" / "pybuild"))
import fastchess
from alphazero.chess_net import ChessNet

ap = argparse.ArgumentParser()
ap.add_argument("--net", default="models/chess_gpu/baseline_67M.pt")
ap.add_argument("--n-games", type=int, default=1024)
ap.add_argument("--sims", type=int, default=64)
ap.add_argument("--runs", type=int, default=3)
ap.add_argument("--compile", action="store_true")
args = ap.parse_args()

DEV = "cuda"
ck = torch.load(ROOT / args.net, map_location=DEV)
net = ChessNet(ck["channels"], ck["blocks"]).to(DEV).eval()
net.load_state_dict(ck["state_dict"])
torch.backends.cudnn.benchmark = True
if args.compile:
    net = torch.compile(net, dynamic=True)   # batch shrinks as games finish -> dynamic
print(f"net {ck['channels']}ch/{ck['blocks']}b | n_games={args.n_games} sims={args.sims} compile={args.compile}", flush=True)


@torch.no_grad()
def eval_fn(planes, lr, lc):
    x = torch.from_numpy(planes).to(DEV, non_blocking=True)
    with torch.autocast("cuda", dtype=torch.float16):
        logits, values = net(x)
    logits = logits.float()
    r = torch.from_numpy(lr).to(DEV); c = torch.from_numpy(lc).to(DEV)
    return (np.ascontiguousarray(logits[r, c].cpu().numpy(), dtype=np.float32),
            np.ascontiguousarray(values.float().cpu().numpy(), dtype=np.float32))


def one_run(seed):
    t = time.time()
    s, _ = fastchess.run_selfplay(eval_fn, args.n_games, args.sims, 16, 50.0, 0.3, 1.5, 120, 3.0, True, seed)
    torch.cuda.synchronize()
    return len(s), time.time() - t


print("warmup (cudnn/compile)...", flush=True)
one_run(1)
ps = []
for i in range(args.runs):
    n, dt = one_run(100 + i)
    ps.append(n / dt)
    print(f"  run {i}: {n} samples / {dt:.1f}s = {n/dt:.0f} pos/s", flush=True)
print(f"\n=== MEDIAN {statistics.median(ps):.0f} pos/s  (compile={args.compile}) ===", flush=True)
