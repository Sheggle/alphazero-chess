"""Evaluate continued-67M checkpoints vs the FROZEN 67M baseline at 2048 sims, L=16.

The clean operating point (all nets ride search monotonically to 2048). Each new
iter_NNNNN.pt (iter > 650 = past the 67M resume) plays baseline_67M.pt over the UHO
suite, both colors, at 2048/L=16. Score -> Elo vs baseline (=0). Idempotent: skips
checkpoints already in elo2048.jsonl. Run concurrently with training (not time-fair-
sensitive — training just adds frames). Builds the Elo-vs-frames curve PAST the plateau.
"""
import sys, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "fastchess" / "pybuild"))
import fastchess
from sweep.arena import _eval_fn_for

SIMS, L, N_OPEN = 2048, 16, 16
CK = ROOT / "models/chess_gpu"
baseline = CK / "baseline_67M.pt"
results = CK / "elo2048.jsonl"

suite = [ln.strip() for ln in (ROOT / "sweep/openings.epd").read_text().splitlines() if ln.strip()]
suite = suite[::max(1, len(suite) // N_OPEN)][:N_OPEN]
GP = 2 * len(suite)

done = set()
if results.exists():
    done = {json.loads(l)["iter"] for l in results.read_text().splitlines() if l.strip()}
iters = sorted(int(p.stem.split("_")[1]) for p in CK.glob("iter_*.pt"))
todo = [it for it in iters if it > 650 and it not in done]
if not todo:
    print("no new checkpoints to eval (latest iter <= 650 or already done)", flush=True)
    sys.exit(0)

efB = _eval_fn_for(baseline, "cuda")
print(f"baseline=67M, {GP} games/pair @ {SIMS} sims L={L}; evaluating {len(todo)} ckpts: {todo}", flush=True)
for it in todo:
    efN = _eval_fn_for(CK / f"iter_{it:05d}.pt", "cuda")
    sc, _ = fastchess.arena_match_openings(efN, efB, suite, 1, 0.0, SIMS, SIMS, L, L,
                                           1.5, 160, 2.0, it, False)
    p = sc / GP
    elo = 400 * np.log10(p / (1 - p)) if 0 < p < 1 else (800.0 if p >= 1 else -800.0)
    fr = it * 66.67 / 650   # ~M frames (iter 650 = 66.67M)
    print(f"  iter {it} (~{fr:.1f}M frames): {sc:.1f}/{GP} ({100*p:.0f}%) -> {elo:+.0f} Elo vs 67M @2048", flush=True)
    with results.open("a") as f:
        f.write(json.dumps({"iter": it, "approx_frames_M": round(fr, 1), "score": sc,
                            "games": GP, "elo": round(float(elo), 1)}) + "\n")
print("done", flush=True)
