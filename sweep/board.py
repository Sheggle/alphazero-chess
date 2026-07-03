"""Print the sweep leaderboard with Elo anchored to the 67M-frame net (= 0)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sweep.arena import fit_elo

s = json.loads((Path(__file__).resolve().parent / "state.json").read_text())
W, N = np.array(s["W"]), np.array(s["N"])
aW, aN = np.array(s.get("aW", []), float), np.array(s.get("aN", []), float)
pool = s["pool"]; n = W.shape[0]
if len(aW) < n:  # anchor catch-up not finished for all
    aW = np.pad(aW, (0, n - len(aW))); aN = np.pad(aN, (0, n - len(aN)))

FW = np.zeros((n + 1, n + 1)); FN = np.zeros((n + 1, n + 1))
FW[:n, :n] = W; FN[:n, :n] = N
FW[n, :n] = aW; FN[n, :n] = aN; FW[:n, n] = aN - aW; FN[:n, n] = aN
R = fit_elo(n + 1, FW, FN); R = R - R[n]               # anchor (index n) = 0

order = np.argsort(-R[:n])
print(f"{'cfg':>3} {'Elo_vs_67M':>10} {'anchor_g':>8}  net    sims n_games   lr      ts  buf")
for j in order:
    c = pool[j]["cfg"]; ag = int(aN[j])
    print(f"{pool[j]['cid']:>3} {R[j]:>+10.0f} {ag:>8}  {c['channels']}c/{c['blocks']}b "
          f"{c['sims']:>3} {c['n_games']:>6}  {c['lr']:.1e} {c['train_steps']:>3} {c['buffer_size']//1000}k")
print(f"  anchor = 67M-frame long-run net = 0 Elo  ({n} configs)")
