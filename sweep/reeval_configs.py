"""Production-faithful recalibrated tournament. Production = max Elo per THINK-TIME,
one game at a time, leaf-parallel. Step 1: calibrate each net's sims/move at the
think-time (single-game, time budget, L=16) -> sims_net (fast nets get more). Step 2:
round-robin at each net's calibrated sims (game-batched only for eval speed; sim count
= production), fixed UHO suite both colors, deterministic best-play. Anchored Elo.
A net that OODs at the sims its speed affords in the think-time is genuinely weak.
"""
import sys, json
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "fastchess" / "pybuild"))
import fastchess
from sweep.arena import _eval_fn_for, fit_elo

THINK_MS = 1000.0     # production think-time per move
L = 16                # leaf-parallel batch (validated ~penalty-free)
N_OPEN = 16
SIM_CAP = 4096        # bound pathological cases

paths = [ROOT / f"sweep/runs/cfg_{i:03d}/final.pt" for i in range(8)] + [ROOT / "models/chess_gpu/latest.pt"]
labels = [f"cfg{i}" for i in range(8)] + ["anchor67M"]
train_sims = [int(torch.load(p, map_location="cpu").get("cfg", {}).get("sims", 64)) for p in paths]
n = len(paths)
efs = [_eval_fn_for(p, "cuda") for p in paths]
suite = [ln.strip() for ln in (ROOT / "sweep/openings.epd").read_text().splitlines() if ln.strip()]
suite = suite[::max(1, len(suite) // N_OPEN)][:N_OPEN]

# 1) calibrate sims/move at the think-time (single-game time mode: ms>0, sims=0)
print(f"calibrating sims/move @ {THINK_MS:.0f}ms (L={L})...", flush=True)
sims = []
for i, ef in enumerate(efs):
    _, st = fastchess.arena_match_openings(ef, ef, suite[:1], 1, THINK_MS, 0, 0, L, L,
                                           1.5, 12, 2.0, i, False)
    s = int(np.mean(list(st["sims_a"]) + list(st["sims_b"])))
    sims.append(int(np.clip(s, L, SIM_CAP)))
    print(f"  {labels[i]:10s} (trained s{train_sims[i]:<3d}) -> {sims[i]} sims/move in {THINK_MS:.0f}ms", flush=True)

# 2) round-robin at per-net calibrated sims, game-batched
print(f"\ntournament: {len(suite)} openings x2 colors = {2*len(suite)} games/pair", flush=True)
W = np.zeros((n, n)); N = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        sc, _ = fastchess.arena_match_openings(efs[i], efs[j], suite, 1, 0.0,
                                               sims[i], sims[j], L, L, 1.5, 160, 2.0,
                                               i * 131 + j, False)
        G = 2 * len(suite)
        W[i, j] += sc; W[j, i] += G - sc; N[i, j] += G; N[j, i] += G
        print(f"  {labels[i]}(s{sims[i]}) vs {labels[j]}(s{sims[j]}): {sc:.1f}/{G}", flush=True)

elo = fit_elo(n, W, N); elo = elo - elo[n - 1]   # anchor = 0
order = np.argsort(-elo)
print("\n=== PRODUCTION leaderboard (1s/move, per-net calibrated sims, UHO both-colors, anchor=0) ===", flush=True)
for k in order:
    print(f"  {labels[k]:10s} train_s{train_sims[k]:<3d} prod_s{sims[k]:<5d} {elo[k]:+7.0f}", flush=True)
(ROOT / "sweep/reeval_results.json").write_text(json.dumps(
    {"labels": labels, "train_sims": train_sims, "prod_sims": sims, "think_ms": THINK_MS,
     "elo": [round(float(x), 1) for x in elo], "W": W.tolist(), "N": N.tolist()}, indent=2))
print("\nwrote sweep/reeval_results.json", flush=True)
