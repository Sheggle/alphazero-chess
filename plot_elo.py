"""Plot Elo-vs-frames from tournament_results.json (produced by tournament_elo.py)."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
d = json.loads((ROOT / "tournament_results.json").read_text())
pl = sorted(d["players"], key=lambda x: x["frames_M"])
f = np.array([p["frames_M"] for p in pl])
elo = np.array([p["elo"] for p in pl])
# shift so the weakest = 0 for a readable "Elo gained" axis
elo = elo - elo.min()

# approx 1-sigma error per player from games played (logistic, ~near 50%)
n_games = np.array([p["games"] for p in pl])
err = 400 / (np.log(10) * np.sqrt(np.maximum(n_games, 1)) * 0.5)

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.errorbar(f, elo, yerr=err, fmt="o-", color="#2266cc", ecolor="#99bbe0",
            capsize=3, lw=2, ms=7, label="checkpoint Elo")
for p, x, y in zip(pl, f, elo):
    ax.annotate(p["label"], (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
ax.set_xlabel("self-play frames (millions)")
ax.set_ylabel("Elo  (weakest checkpoint = 0)")
ax.set_title(f"AlphaZero chess: Elo vs frames  (round-robin, {d['g_per_pair']} games/pair, "
             f"{d['sims']} sims)\nspan {elo.max()-elo.min():.0f} Elo over {f.min():.0f}->{f.max():.0f}M frames")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
out = ROOT / "elo_vs_frames.png"
fig.savefig(out, dpi=130)
print(f"saved {out}")
for p in pl:
    print(f"  {p['label']:5s} {p['frames_M']:5.1f}M -> {p['elo']-elo.min()*0:+.0f}  (raw {p['elo']:+.1f})")
