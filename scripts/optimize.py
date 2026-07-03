"""Run a single low-sim training experiment and report both constraints.

Usage: PYTHONPATH=. uv run python scripts/optimize.py <name> [iterations]
Edit `make_cfg` to change the experiment. Prints the opt(mcts) curve, the first
iteration reaching >=0.90 (K), and the min opt(mcts) over [K, 3K] and [K, end].
"""
import os
import sys

from alphazero.train import Config, train


def _f(name, default):
    return float(os.environ.get(name, default))

THRESH = 0.90


def analyze(log, name):
    curve = [r["opt_rate_mcts"] for r in log]
    crossings = [r["iter"] for r in log if r["opt_rate_mcts"] >= THRESH]
    print(f"\n=== {name} ===")
    print("opt(mcts) curve:", " ".join(f"{c:.3f}" for c in curve))
    peak = max(curve)
    print(f"peak opt(mcts): {peak:.4f}")
    if not crossings:
        print(f"NEVER reached {THRESH}. FAIL constraint 1.")
        return {"name": name, "K": None, "peak": peak, "pass1": False, "pass2": False}
    K = crossings[0]
    end = log[-1]["iter"]
    min_to_end = min(r["opt_rate_mcts"] for r in log if r["iter"] >= K)
    target_3k = 3 * K
    covered = end >= target_3k
    min_to_3k = min((r["opt_rate_mcts"] for r in log if K <= r["iter"] <= target_3k), default=peak)
    pass1 = peak >= THRESH
    pass2 = covered and min_to_3k >= THRESH
    print(f"K (first >= {THRESH}): iter {K}   -> need stable through 3K = {target_3k} (ran to {end})")
    print(f"min opt(mcts) over [K..end]: {min_to_end:.4f}")
    print(f"min opt(mcts) over [K..3K]:  {min_to_3k:.4f}  {'(3K not reached)' if not covered else ''}")
    print(f"CONSTRAINT 1 (>=0.90 at 3 sims): {'PASS' if pass1 else 'FAIL'}")
    print(f"CONSTRAINT 2 (stable to 3K):     {'PASS' if pass2 else 'FAIL (or 3K not reached)'}")
    return {"name": name, "K": K, "peak": peak, "min_to_3k": min_to_3k,
            "min_to_end": min_to_end, "pass1": pass1, "pass2": pass2}


def make_cfg(iterations):
    # Gumbel acting + completed-Q target, sims=3 everywhere. Stability knobs are
    # overridable via env vars for sweeping (ENTROPY, C_SCALE, LR, WD, SP_SIMS).
    sims = int(_f("SP_SIMS", 3))
    return Config(
        iterations=iterations,
        games_per_iter=40,
        selfplay_sims=sims,
        eval_sims=sims,
        use_gumbel=True,
        gumbel_max_considered=8,
        c_visit=50.0,
        c_scale=_f("C_SCALE", 1.0),
        c_puct=1.5,
        train_steps=150,
        batch_size=256,
        lr=_f("LR", 1e-3),
        weight_decay=_f("WD", 1e-4),
        entropy_coef=_f("ENTROPY", 0.0),
        eval_mcts_full=True,
        eval_random_games=100,
        seed=0,
    )


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "exp"
    iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    cfg = make_cfg(iterations)
    _, _, cfg = train(cfg, verbose=True)
    analyze(cfg.log, name)


if __name__ == "__main__":
    main()
