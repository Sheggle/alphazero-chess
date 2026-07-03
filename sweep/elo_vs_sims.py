"""CLEAN value-head search-capability test: same net, different sim counts, head-to-head.

For each net, treat each sim count as a "player" and play a round-robin: net@s_i vs
net@s_j over the fixed UHO suite, both colors, per-side sims (identical weights both
sides -> the ONLY variable is search depth). Fit Elo, anchored at the lowest sim count.
Curve shape = the answer: monotone-up = search helps (saturating); rise-then-FALL =
value head actively misleads deeper search (OOD). L fixed across all points so the
leaf-parallel offset is ~constant and doesn't confound the sims axis.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "fastchess" / "pybuild"))
import fastchess
from sweep.arena import _eval_fn_for, fit_elo

L, N_OPEN = 16, 16
SIMS = [64, 128, 256, 512, 1024, 2048]
NETS = [("cfg1", "sweep/runs/cfg_001/final.pt"), ("cfg6", "sweep/runs/cfg_006/final.pt")]  # sims=8 nets: does the old "40%->2.5%" collapse reproduce in the clean arena?

suite = [ln.strip() for ln in (ROOT / "sweep/openings.epd").read_text().splitlines() if ln.strip()]
suite = suite[::max(1, len(suite) // N_OPEN)][:N_OPEN]
GP = 2 * len(suite)
n = len(SIMS)

for name, path in NETS:
    ef = _eval_fn_for(ROOT / path, "cuda")
    W = np.zeros((n, n)); N = np.zeros((n, n))
    print(f"\n### {name}: same net, {GP} games/pair, L={L}", flush=True)
    for i in range(n):
        for j in range(i + 1, n):
            sc, _ = fastchess.arena_match_openings(ef, ef, suite, 1, 0.0, SIMS[i], SIMS[j],
                                                   L, L, 1.5, 160, 2.0, i * 100 + j, False)
            W[i, j] += sc; W[j, i] += GP - sc; N[i, j] += GP; N[j, i] += GP
            print(f"  @{SIMS[i]:5d} vs @{SIMS[j]:5d}: {sc:.1f}/{GP}  ({'higher-sims wins' if sc < GP/2 else 'lower-sims wins'})", flush=True)
    elo = fit_elo(n, W, N); elo = elo - elo[0]   # anchor curve at lowest sims = 0
    peak = SIMS[int(np.argmax(elo))]
    print(f"=== {name}: Elo vs sims (vs @{SIMS[0]}=0) | peak @ {peak} sims ===", flush=True)
    for k in range(n):
        print(f"  @{SIMS[k]:5d} -> {elo[k]:+6.0f}", flush=True)
