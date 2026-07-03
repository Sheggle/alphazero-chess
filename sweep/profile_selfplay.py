"""Profile run_selfplay: where does each round go, and how much is the GPU idle?
Clean run -> true ms/round; instrumented run (with syncs) -> accurate per-phase ms
(H2D, forward, D2H+gather+sync). GPU-busy% = forward_ms / clean_round_ms. Both modes
(refill=False/True) to isolate the refill per-completion overhead.
"""
import sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "fastchess" / "pybuild"))
import fastchess
from alphazero.chess_net import ChessNet

ck = torch.load(ROOT / "models/chess_gpu/baseline_67M.pt", map_location="cuda")
net = ChessNet(ck["channels"], ck["blocks"]).cuda().eval(); net.load_state_dict(ck["state_dict"])
torch.backends.cudnn.benchmark = True
NG, SIMS, MP = 1024, 64, 120

rounds = [0]
@torch.no_grad()
def ef_clean(planes, lr, lc):
    rounds[0] += 1
    x = torch.from_numpy(planes).to("cuda", non_blocking=True)
    with torch.autocast("cuda", dtype=torch.float16):
        logits, values = net(x)
    logits = logits.float()
    r = torch.from_numpy(lr).to("cuda"); c = torch.from_numpy(lc).to("cuda")
    return (np.ascontiguousarray(logits[r, c].cpu().numpy(), dtype=np.float32),
            np.ascontiguousarray(values.float().cpu().numpy(), dtype=np.float32))

T = {}
@torch.no_grad()
def ef_prof(planes, lr, lc):
    t0 = time.perf_counter()
    x = torch.from_numpy(planes).to("cuda", non_blocking=True)
    torch.cuda.synchronize(); t1 = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.float16):
        logits, values = net(x)
    torch.cuda.synchronize(); t2 = time.perf_counter()
    logits = logits.float()
    r = torch.from_numpy(lr).to("cuda"); c = torch.from_numpy(lc).to("cuda")
    ll = np.ascontiguousarray(logits[r, c].cpu().numpy(), dtype=np.float32)
    vv = np.ascontiguousarray(values.float().cpu().numpy(), dtype=np.float32)
    torch.cuda.synchronize(); t3 = time.perf_counter()
    for k, dv in (("h2d", t1-t0), ("fwd", t2-t1), ("d2h", t3-t2)):
        T[k] = T.get(k, 0.0) + dv
    T["n"] = T.get("n", 0) + 1
    return ll, vv

def run(refill, ef):
    return fastchess.run_selfplay(ef, NG, SIMS, 16, 50., 0.3, 1.5, MP, 3., True, 7, refill)

print("warmup...", flush=True); run(False, ef_clean)
for refill in (False, True):
    rounds[0] = 0; t = time.perf_counter(); s, _ = run(refill, ef_clean); tot = time.perf_counter() - t
    rms = 1000 * tot / rounds[0]
    print(f"\n=== refill={refill} ===", flush=True)
    print(f"CLEAN: {len(s)} samples / {tot:.1f}s = {len(s)/tot:.0f} pos/s | {rounds[0]} rounds | {rms:.2f} ms/round", flush=True)
    T.clear(); run(refill, ef_prof); n = T["n"]
    fwd = 1000*T["fwd"]/n; h2d = 1000*T["h2d"]/n; d2h = 1000*T["d2h"]/n
    print(f"PHASES (ms/round, sync-attributed): forward {fwd:.2f} | H2D {h2d:.2f} | D2H+gather+sync {d2h:.2f} | eval_fn {fwd+h2d+d2h:.2f}", flush=True)
    print(f"GPU-BUSY = forward/clean-round = {100*fwd/rms:.0f}%  ->  GPU-IDLE {100*(rms-fwd)/rms:.0f}% ({rms-fwd:.2f} ms/round on CPU/transfers/tree)", flush=True)
