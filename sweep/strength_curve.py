"""High-sim strength-vs-L curve for the Rust PUCT arena. Now that the search is
Rust (thousands of sims/move are cheap), test whether the leaf-parallel (virtual
loss) penalty AMORTIZES as the round count grows: play L=1 (side A) vs L=k (side
B) with the SAME net at EQUAL fixed sims, across sim budgets. score(L=1 side)
near 50% means L=k has caught up = the L penalty has amortized (high L ~free).

Uses the same net both sides, so value-head sim-sensitivity cancels and the score
isolates the leaf-parallel approximation. Writes to --out.
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (str(ROOT), str(ROOT / "fastchess" / "pybuild")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fastchess  # noqa: E402
from sweep.batched_arena import load_evaluator, make_eval_fn  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--net", default=str(ROOT / "models" / "chess_gpu" / "iter_00600.pt"))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-ply", type=int, default=40)
    ap.add_argument("--sims", default="128,512,2048")
    ap.add_argument("--ls", default="16,64,256")
    ap.add_argument("--open-plies", type=int, default=6)  # opening randomization -> distinct games
    args = ap.parse_args()

    ev = load_evaluator(args.net, "cuda")
    ef = make_eval_fn(ev, fp16=True)
    sims_list = [int(x) for x in args.sims.split(",")]
    ls = [int(x) for x in args.ls.split(",")]
    lines = [f"### STRENGTH vs L across sim budgets (net={Path(args.net).name}, "
             f"N={args.n}, max_ply={args.max_ply})",
             "  L=1(A) vs L=k(B), same net, equal fixed sims. score is L=1 side; "
             "->50% = L penalty amortized."]
    for lb in ls:
        cells = []
        for s in sims_list:
            sc, st = fastchess.arena_match(ef, ef, args.n, 0.0, s, s, 1, lb, 1.5,
                                           args.max_ply, 2.0, args.open_plies, 5, False)
            a = np.mean(st["sims_a"]); b = np.mean(st["sims_b"])
            cells.append(f"S={s:5d}: L1={100*sc/args.n:5.1f}% (simsA={a:.0f} simsB={b:.0f})")
        lines.append(f"  L=1 vs L={lb:3d} | " + " | ".join(cells))
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
