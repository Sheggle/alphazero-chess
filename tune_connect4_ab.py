"""Tune the line weights W1,W2,W3 for the Score-Four alpha-beta bot.

Method: at a FIXED shallow depth (so the heuristic decides most games), run a
round-robin among candidate weight-triples. Each pair plays every opening from
BOTH colours; score = points (win 1 / draw 0.5 / loss 0). W3 is fixed to 1.0
(weights are scale-invariant) so only W1,W2 are searched: coarse grid ->
hill-climb refine -> sensitivity report.

Run:  PYTHONPATH=. uv run python tune_connect4_ab.py
"""
from __future__ import annotations

import itertools
import multiprocessing as mp
import random
from concurrent.futures import ProcessPoolExecutor

# macOS defaults to 'spawn', which deadlocks/breaks worker imports for this
# script; 'fork' is safe here (pure-numpy compute, no torch/GUI in workers).
try:
    mp.set_start_method("fork")
except RuntimeError:
    pass

from alphazero.connect4_env import Connect4Game
from alphazero import connect4_ab as ab

DEPTH = 4
N_WORKERS = 8


def make_openings(n: int, plies: int = 4, seed: int = 12345) -> list[tuple[int, ...]]:
    """n diverse, non-terminal opening move sequences of `plies` half-moves."""
    rng = random.Random(seed)
    outs: set[tuple[int, ...]] = set()
    tries = 0
    while len(outs) < n and tries < n * 200:
        tries += 1
        g = Connect4Game()
        seq = []
        ok = True
        for _ in range(plies):
            if g.is_terminal():
                ok = False
                break
            m = rng.choice(g.legal_moves())
            seq.append(m)
            g = g.apply(m)
        if ok and not g.is_terminal():
            outs.add(tuple(seq))
    return sorted(outs)


def play_game(wA, wB, opening, depth=DEPTH) -> int:
    """A as player +1, B as player -1. Returns env result (+1 A wins / -1 B / 0)."""
    ea = ab.Connect4AB(tuple(wA), use_tt=False)
    eb = ab.Connect4AB(tuple(wB), use_tt=False)
    g = Connect4Game()
    for m in opening:
        if g.is_terminal():
            break
        g = g.apply(m)
    while not g.is_terminal():
        eng = ea if g.to_play == 1 else eb
        col, _ = eng.best_move_depth(g, depth)
        g = g.apply(col)
    return g.result()


def _job(args):
    wA, wB, opening, depth = args
    return play_game(wA, wB, opening, depth)


def round_robin(cands, openings, depth=DEPTH, ex=None):
    """Return dict cand-index -> total points over the full double-colour RR."""
    jobs = []
    meta = []  # (i, j) for A=i vs B=j
    for i, j in itertools.permutations(range(len(cands)), 2):
        # i as +1 vs j as -1, for every opening
        for op in openings:
            jobs.append((cands[i], cands[j], op, depth))
            meta.append((i, j))
    results = list(ex.map(_job, jobs, chunksize=4))
    pts = [0.0] * len(cands)
    games = [0] * len(cands)
    for (i, j), r in zip(meta, results):
        games[i] += 1
        games[j] += 1
        if r == 1:
            pts[i] += 1.0
        elif r == -1:
            pts[j] += 1.0
        else:
            pts[i] += 0.5
            pts[j] += 0.5
    return pts, games


def main():
    openings = make_openings(8)
    print(f"{len(openings)} openings, depth {DEPTH}\n")

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        # ---- coarse grid ----
        w1s = [0.03, 0.07, 0.12, 0.20, 0.35]
        w2s = [0.15, 0.30, 0.45, 0.60]
        coarse = [(w1, w2, 1.0) for w1 in w1s for w2 in w2s if w1 < w2]
        print(f"COARSE grid: {len(coarse)} candidates, "
              f"{len(coarse)*(len(coarse)-1)*len(openings)} games")
        pts, games = round_robin(coarse, openings, ex=ex)
        ranked = sorted(range(len(coarse)), key=lambda i: -pts[i])
        print("  rank  W1     W2     pts   /games")
        for r, i in enumerate(ranked):
            print(f"  {r+1:>3}  {coarse[i][0]:.3f}  {coarse[i][1]:.3f}  "
                  f"{pts[i]:5.1f} /{games[i]}")
        best = coarse[ranked[0]]
        print(f"\ncoarse best: W1={best[0]}, W2={best[1]}, W3=1.0\n")

        # ---- fine grid around the coarse winner ----
        b1, b2 = best[0], best[1]
        f1 = sorted({round(max(0.01, b1 * f), 4) for f in (0.6, 0.8, 1.0, 1.25, 1.6)})
        f2 = sorted({round(b2 * f, 4) for f in (0.7, 0.85, 1.0, 1.2, 1.4)})
        fine = [(w1, w2, 1.0) for w1 in f1 for w2 in f2 if w1 < w2]
        # keep the current default in the mix for reference
        if ab.DEFAULT_WEIGHTS not in fine:
            fine.append(ab.DEFAULT_WEIGHTS)
        print(f"FINE grid: {len(fine)} candidates, "
              f"{len(fine)*(len(fine)-1)*len(openings)} games")
        pts, games = round_robin(fine, openings, ex=ex)
        ranked = sorted(range(len(fine)), key=lambda i: -pts[i])
        print("  rank  W1     W2     pts   /games")
        for r, i in enumerate(ranked[:12]):
            print(f"  {r+1:>3}  {fine[i][0]:.3f}  {fine[i][1]:.3f}  "
                  f"{pts[i]:5.1f} /{games[i]}")
        winner = fine[ranked[0]]
        print(f"\n*** TUNED WEIGHTS: W1={winner[0]}, W2={winner[1]}, W3=1.0 ***\n")

        # ---- sensitivity: winner vs perturbed copies of itself (gauntlet) ----
        print("SENSITIVITY (winner as reference; each perturbation plays winner "
              "both colours, all openings):")
        perts = []
        labels = []
        for name, w in [
            ("winner", winner),
            ("W1 x0.5", (winner[0] * 0.5, winner[1], 1.0)),
            ("W1 x2.0", (winner[0] * 2.0, winner[1], 1.0)),
            ("W2 x0.6", (winner[0], winner[1] * 0.6, 1.0)),
            ("W2 x1.6", (winner[0], winner[1] * 1.6, 1.0)),
            ("flat 1,1,1", (1.0, 1.0, 1.0)),
            ("W2=W3 (0,1,1)", (0.001, 1.0, 1.0)),
        ]:
            perts.append(w)
            labels.append(name)
        # gauntlet: perturbation p vs winner, both colours, all openings
        jobs = []
        meta = []
        for k, w in enumerate(perts):
            for op in openings:
                jobs.append((w, winner, op, DEPTH)); meta.append((k, 0))
                jobs.append((winner, w, op, DEPTH)); meta.append((k, 1))
        res = list(ex.map(_job, jobs, chunksize=4))
        score = [0.0] * len(perts)
        tot = [0] * len(perts)
        for (k, side), r in zip(meta, res):
            tot[k] += 1
            if side == 0:      # perturbation is +1
                score[k] += 1.0 if r == 1 else (0.5 if r == 0 else 0.0)
            else:              # perturbation is -1
                score[k] += 1.0 if r == -1 else (0.5 if r == 0 else 0.0)
        print("  perturbation      score vs winner")
        for k, name in enumerate(labels):
            print(f"  {name:<16}  {score[k]:.1f}/{tot[k]}  "
                  f"({100*score[k]/tot[k]:.0f}%)")


if __name__ == "__main__":
    main()
