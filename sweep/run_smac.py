"""SMAC sweep v2 — production-eval objective, varied-sims search space, warm-started.

Objective = Elo at PRODUCTION think-time: each net is calibrated to its sims/move at
THINK_MS (single-game, leaf-parallel), then plays the pool (incl the 67M anchor=0) at
those per-net sims over the fixed UHO suite, both colors. Search space adds varied
training sims (sims_lo..sims_hi, sampled per iter in train_config) so the value head
isn't brittle at one depth. Warm-starts the pool + SMAC surrogate from the 8 v1 configs
(sweep/reeval_results.json).

RUN SEQUENTIALLY ON A CLEAN BOX — nothing else on the GPU, or the 1h training budgets
(and the calibration) are no longer time-fair.
  env: TIME_BUDGET=3600  N_TRIALS=60  EVAL_OPP=5  THINK_MS=1000
"""
import json, os, sys, subprocess, random
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "fastchess" / "pybuild"))
import fastchess
from sweep.arena import _eval_fn_for, fit_elo
from ConfigSpace import ConfigurationSpace, Configuration, Float, Integer, Categorical
from smac import HyperparameterOptimizationFacade as HPO, Scenario
from smac.runhistory.dataclasses import TrialValue, TrialInfo

BUDGET = float(os.environ.get("TIME_BUDGET", 3600))
N_TRIALS = int(os.environ.get("N_TRIALS", 60))
EVAL_OPP = int(os.environ.get("EVAL_OPP", 5))
THINK_MS = float(os.environ.get("THINK_MS", 1000))
L_EVAL, N_OPEN = 16, 16
DEV = "cuda"
SDIR = ROOT / "sweep"; RUNS = SDIR / "runs"; STATE = SDIR / "state_v2.json"
ANCHOR_PATH = str(ROOT / "models/chess_gpu/latest.pt")
SUITE = [ln.strip() for ln in (SDIR / "openings.epd").read_text().splitlines() if ln.strip()]
SUITE = SUITE[::max(1, len(SUITE) // N_OPEN)][:N_OPEN]
GP = 2 * len(SUITE)   # games per pair (each opening both colors)


def space():
    cs = ConfigurationSpace(seed=0)
    cs.add([
        Float("lr", (1e-4, 5e-3), log=True, default=1e-3),
        Float("lr_final_frac", (0.03, 1.0), log=True, default=0.3),
        Categorical("channels", [48, 64, 96, 128], default=64),
        Categorical("blocks", [4, 6, 8, 10], default=6),
        Integer("buffer_size", (50_000, 1_000_000), log=True, default=300_000),
        Integer("train_steps", (25, 400), log=True, default=100),
        Categorical("batch_size", [256, 512, 1024, 2048], default=512),
        Float("entropy_coef", (0.0, 0.2), default=0.05),
        Float("c_scale", (0.1, 1.5), default=0.3),
        Categorical("mc", [4, 8, 16, 32], default=16),
        Categorical("n_games", [256, 512, 1024, 2048], default=512),
        Integer("mat_thresh", (1, 5), default=3),
        Integer("sims_lo", (8, 256), log=True, default=16),   # varied training sims:
        Integer("sims_hi", (8, 256), log=True, default=64),   # each iter ~ loguniform[lo,hi]
    ])
    return cs


def calibrate(ef):
    """sims/move this net reaches at THINK_MS, single game, leaf-parallel."""
    _, st = fastchess.arena_match_openings(ef, ef, SUITE[:1], 1, THINK_MS, 0, 0, L_EVAL, L_EVAL,
                                           1.5, 12, 2.0, 0, False)
    return int(np.clip(np.mean(list(st["sims_a"]) + list(st["sims_b"])), L_EVAL, 4096))


def play(efa, efb, sa, sb, seed):
    sc, _ = fastchess.arena_match_openings(efa, efb, SUITE, 1, 0.0, sa, sb, L_EVAL, L_EVAL,
                                           1.5, 160, 2.0, seed, False)
    return sc


def train(cfg, cid):
    rdir = RUNS / f"v2_{cid:03d}"; rdir.mkdir(parents=True, exist_ok=True)
    full = dict(cfg); full.update(run_dir=str(rdir), time_budget_s=BUDGET, seed=cid + 1, weight_decay=1e-4)
    (rdir / "config.json").write_text(json.dumps(full))
    env = dict(os.environ, PYTHONPATH=f"{ROOT}:{ROOT}/fastchess/pybuild")
    print(f"[train] v2 {cid}: {dict(cfg)}", flush=True)
    subprocess.run([sys.executable, str(SDIR / "train_config.py"), str(rdir / "config.json")],
                   cwd=str(ROOT), env=env, check=True)
    return str(rdir / "final.pt")


def grow(M, n):
    o = np.zeros((n, n)); o[:M.shape[0], :M.shape[1]] = M; return o


def save_state(pool, W, N, trials, ai):
    STATE.write_text(json.dumps({"pool": [{k: v for k, v in p.items() if k != "ef"} for p in pool],
                                 "W": W.tolist(), "N": N.tolist(), "trials": trials, "anchor_idx": ai}))


def map_to_space(cs, old):
    s = int(old["sims"])
    return Configuration(cs, values=dict(
        lr=float(old["lr"]), lr_final_frac=float(old["lr_final_frac"]), channels=int(old["channels"]),
        blocks=int(old["blocks"]), buffer_size=int(old["buffer_size"]), train_steps=int(old["train_steps"]),
        batch_size=int(old["batch_size"]), entropy_coef=float(old["entropy_coef"]), c_scale=float(old["c_scale"]),
        mc=int(old["mc"]), n_games=int(old["n_games"]), mat_thresh=int(old["mat_thresh"]), sims_lo=s, sims_hi=s))


def warmstart(cs, smac):
    """Pool + W/N + SMAC surrogate from the 8 v1 configs (reeval_results.json)."""
    r = json.loads((SDIR / "reeval_results.json").read_text())
    pool = []
    for i in range(8):
        cj = json.loads((RUNS / f"cfg_{i:03d}/config.json").read_text())
        pool.append({"cid": i, "cfg": cj, "path": str(RUNS / f"cfg_{i:03d}/final.pt"), "prod_sims": r["prod_sims"][i]})
        try:
            smac.tell(TrialInfo(map_to_space(cs, cj), seed=0), TrialValue(cost=-float(r["elo"][i])))
        except Exception as e:
            print(f"  warmstart tell cfg{i} skipped: {e}", flush=True)
    pool.append({"cid": -1, "cfg": None, "path": ANCHOR_PATH, "prod_sims": r["prod_sims"][8], "anchor": True})
    trials = [{"cid": i, "elo": r["elo"][i]} for i in range(8)]
    return pool, np.array(r["W"], float), np.array(r["N"], float), trials, 8


def main():
    cs = space()
    scenario = Scenario(cs, n_trials=N_TRIALS, deterministic=True, output_directory=SDIR / "smac_v2")
    fresh = not STATE.exists()
    smac = HPO(scenario, lambda config, seed=0: 0.0, overwrite=fresh)
    if fresh:
        pool, W, N, trials, ai = warmstart(cs, smac)
        save_state(pool, W, N, trials, ai)
    else:
        s = json.loads(STATE.read_text())
        pool, W, N, trials, ai = s["pool"], np.array(s["W"]), np.array(s["N"]), s["trials"], s["anchor_idx"]
    for p in pool:
        p["ef"] = _eval_fn_for(p["path"], DEV)
    print(f"sweep v2: budget {BUDGET:.0f}s, think {THINK_MS:.0f}ms, {len(SUITE)} openings, "
          f"pool={len(pool)} ({len(trials)} warm trials), eval vs {EVAL_OPP}+anchor", flush=True)

    while len(trials) < N_TRIALS:
        info = smac.ask()
        cfg = {k: (v.item() if hasattr(v, "item") else v) for k, v in dict(info.config).items()}
        cid = max([t["cid"] for t in trials] + [7]) + 1
        try:
            path = train(cfg, cid)
        except subprocess.CalledProcessError as e:
            print(f"[train] v2 {cid} FAILED: {e}", flush=True)
            smac.tell(info, TrialValue(cost=1e6)); trials.append({"cid": cid, "failed": True}); continue
        ef = _eval_fn_for(path, DEV); psims = calibrate(ef); new = len(pool)
        pool.append({"cid": cid, "cfg": cfg, "path": path, "prod_sims": psims, "ef": ef})
        W, N = grow(W, len(pool)), grow(N, len(pool))
        opp = [ai] + random.sample([k for k in range(new) if k != ai],
                                   min(EVAL_OPP - 1, new - 1))
        for o in opp:
            sc = play(ef, pool[o]["ef"], psims, pool[o]["prod_sims"], cid * 1000 + o)
            W[new, o] += sc; W[o, new] += GP - sc; N[new, o] += GP; N[o, new] += GP
            print(f"  [eval] v2_{cid}(s{psims}) vs {pool[o].get('cid')}(s{pool[o]['prod_sims']}): {sc:.1f}/{GP}", flush=True)
        elo = fit_elo(len(pool), W, N); elo = elo - elo[ai]
        smac.tell(info, TrialValue(cost=-float(elo[new])))
        trials.append({"cid": cid, "cfg": cfg, "prod_sims": psims, "elo": round(float(elo[new]), 1)})
        save_state(pool, W, N, trials, ai)
        order = np.argsort(-elo)
        print(f"=== after v2 {cid}: {elo[new]:+.0f} vs anchor | top: " +
              ", ".join(f"{pool[j].get('cid')}={elo[j]:+.0f}" for j in order[:6]) + "\n", flush=True)
    print("SWEEP v2 DONE", flush=True)


if __name__ == "__main__":
    main()
