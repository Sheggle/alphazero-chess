"""Roofline vs actual self-play GPU utilization.

Roofline = forward the net on a large PRECOMPUTED on-GPU batch in a tight loop
(no CPU, no transfers in the timed region) -> max inferences/s the model can do.
Actual = run real self-play with an instrumented eval_fn that accumulates GPU
time and inference count. True util = actual_inf/s (over wall) / roofline_inf/s.
"""
import sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "fastchess" / "pybuild"))
import fastchess
from alphazero.chess_net import ChessNet

dev = "cuda"
ck = torch.load(ROOT / "models/chess_gpu/latest.pt", map_location=dev)
net = ChessNet(ck["channels"], ck["blocks"]).to(dev).eval()
net.load_state_dict(ck["state_dict"])
torch.backends.cudnn.benchmark = True


@torch.no_grad()
def fwd(x):
    with torch.autocast("cuda", dtype=torch.float16):
        return net(x)


print(f"net {ck['channels']}ch/{ck['blocks']}b, {sum(p.numel() for p in net.parameters())/1e6:.1f}M params\n")
print("ROOFLINE (precomputed on-GPU input, tight fwd loop, fp16, no transfers):")
roof = {}
for B in [256, 512, 1024, 2048, 4096]:
    x = torch.randn(B, 18, 8, 8, device=dev)
    for _ in range(8):
        fwd(x)
    torch.cuda.synchronize()
    t = time.time(); n = 60
    for _ in range(n):
        fwd(x)
    torch.cuda.synchronize()
    dt = time.time() - t
    roof[B] = B * n / dt
    print(f"  B={B:5d}: {roof[B]/1e3:8.1f}k inf/s   ({1e3*dt/n:6.2f} ms/fwd)")

# ---- actual self-play, instrumented ----
ev_t = [0.0]; ev_n = [0]; ev_calls = [0]


@torch.no_grad()
def eval_fn(planes, lr, lc):
    t = time.time()
    x = torch.from_numpy(planes).to(dev, non_blocking=True)
    with torch.autocast("cuda", dtype=torch.float16):
        logits, values = net(x)
    logits = logits.float()
    r = torch.from_numpy(lr).to(dev); c = torch.from_numpy(lc).to(dev)
    ll = np.ascontiguousarray(logits[r, c].cpu().numpy(), dtype=np.float32)
    vv = np.ascontiguousarray(values.float().cpu().numpy(), dtype=np.float32)
    torch.cuda.synchronize()
    ev_t[0] += time.time() - t; ev_n[0] += planes.shape[0]; ev_calls[0] += 1
    return ll, vv


print("\nwarming cudnn for self-play batch sizes (discarded)...")
fastchess.run_selfplay(eval_fn, 1024, 16, 16, 50.0, 0.3, 1.5, 120, 3.0, True, 1)
ev_t[0] = 0.0; ev_n[0] = 0; ev_calls[0] = 0  # reset after warmup

print("ACTUAL self-play (n_games=1024, 64 sims), warmed:")
t = time.time()
samples, stats = fastchess.run_selfplay(eval_fn, 1024, 64, 16, 50.0, 0.3, 1.5, 120, 3.0, True, 777)
sp = time.time() - t
avg_B = ev_n[0] / ev_calls[0]
act_over_wall = ev_n[0] / sp
act_in_eval = ev_n[0] / ev_t[0]
# roofline at the avg batch size actually used (interp between measured points)
import bisect
Bs = sorted(roof)
i = min(bisect.bisect_left(Bs, avg_B), len(Bs) - 1)
roof_at_avgB = roof[Bs[i]]
print(f"  wall {sp:.1f}s | rounds(eval calls) {ev_calls[0]} | avg batch {avg_B:.0f} | positions {len(samples)}")
print(f"  total inferences: {ev_n[0]/1e6:.2f}M")
print(f"  time in eval_fn (GPU fwd+gather+D2H): {ev_t[0]:.1f}s = {100*ev_t[0]/sp:.0f}% of wall")
print(f"  time in Rust tree ops (CPU):          {sp-ev_t[0]:.1f}s = {100*(sp-ev_t[0])/sp:.0f}% of wall")
print(f"\n  actual inf/s OVER WALL:   {act_over_wall/1e3:7.1f}k")
print(f"  actual inf/s IN eval_fn:  {act_in_eval/1e3:7.1f}k")
print(f"  roofline @ B~{Bs[i]}:        {roof_at_avgB/1e3:7.1f}k")
print(f"\n  >>> % OF ROOFLINE (over wall):    {100*act_over_wall/roof_at_avgB:.1f}%")
print(f"  >>> % OF ROOFLINE (in-eval only): {100*act_in_eval/roof_at_avgB:.1f}%")
