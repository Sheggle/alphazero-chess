"""Place a sweep config on the 67M run's Elo-vs-frames curve by playing it directly
against the long-run checkpoints (same conditions as the original tournament)."""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "fastchess" / "pybuild"))
from tournament_elo import load_ev, play_game, fit_elo, PLAYERS   # reuse (16-sim games)

CFG = sys.argv[1] if len(sys.argv) > 1 else "sweep/runs/cfg_004/final.pt"
CFG_FRAMES = float(sys.argv[2]) if len(sys.argv) > 2 else 7.5
G = 14

d = json.loads((ROOT / "tournament_results.json").read_text())
players = d["players"]; W = np.array(d["W"]); N = np.array(d["N"]); n = len(players)
ev_cfg = load_ev(ROOT / CFG)
evs = [load_ev(ROOT / PLAYERS[i][1]) for i in range(n)]
W2 = np.zeros((n + 1, n + 1)); N2 = np.zeros((n + 1, n + 1)); W2[:n, :n] = W; N2[:n, :n] = N
for i in range(n):
    sa = 0.0
    for k in range(G):
        rng = np.random.default_rng(4000 + i * 17 + k)
        if k % 2 == 0:
            r = play_game(ev_cfg, evs[i], rng); sa += 1 if r > 0 else (0.5 if r == 0 else 0)
        else:
            r = play_game(evs[i], ev_cfg, rng); sa += 1 if r < 0 else (0.5 if r == 0 else 0)
    W2[n, i] = sa; W2[i, n] = G - sa; N2[n, i] = G; N2[i, n] = G
    print(f"  cfg vs {players[i]['label']} ({players[i]['frames_M']:.1f}M): {sa:.1f}/{G}", flush=True)
R = fit_elo(n + 1, W2, N2); R = R - R[:n].mean()   # center on checkpoint mean (tournament frame)
print("\nElo-vs-frames curve (67M run) with the config placed:")
for i in sorted(range(n), key=lambda i: players[i]["frames_M"]):
    print(f"  {players[i]['frames_M']:5.1f}M -> {R[i]:+6.0f}")
print(f"\n  CONFIG ({CFG_FRAMES:.1f}M actual frames, 1h) -> {R[n]:+.0f} Elo")
# interpolate frame-equivalent
order = sorted(range(n), key=lambda i: R[i]); fr = [players[i]["frames_M"] for i in order]; el = [R[i] for i in order]
eq = np.interp(R[n], el, fr)
print(f"  => equivalent to ~{eq:.1f}M frames of the 67M run")
