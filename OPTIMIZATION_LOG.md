# Optimization log — low-sim AlphaZero for tic-tac-toe

**Goal (set 2026-06-26).** Two constraints:
1. **Few sims.** opt-rate ≥ 0.90 with **self-play sims = 3 AND eval sims = 3**.
2. **Overtraining stability.** If iter *K* is when opt-rate first reaches 0.90,
   training to **3K** must never dip below 0.90 or collapse.

**Metric.** `opt(mcts)` = optimal-move rate (vs exact solver) of the AZ agent,
evaluated at **eval_sims = 3** over **all 4520** non-terminal states (full set,
not a sample — at 3 sims this is cheap and removes metric noise, so a real dip is
distinguishable from jitter). Also tracked: `opt(raw)` (pure policy, no search).

The "stability score" of a run = `min(opt(mcts))` over all iterations from K to
3K. A run passes constraint 2 iff that stays ≥ 0.90.

---

## Baseline measurements (before any change)

Existing net (trained at 100 sims, `models/ttt_az.pt`), opt-rate vs eval sims:

| eval sims | raw | 1 | 2 | 3 | 5 | 10 | 25 | 100 |
|-----------|-----|---|---|---|---|----|----|-----|
| opt-rate  | 0.950 | 0.587 | 0.587 | 0.945 | 0.973 | 0.984 | 0.988 | 0.992 |

The 0.587 at 1–2 sims with a sharp jump to 0.945 at 3 is the symptom of a bug,
not normal behavior (see Change 1).

---

## Change 1 — fix PUCT root-visit init (the 1–2 sim cliff)

**Problem.** `AZMCTS._expand(root)` left `root.n = 0`. On simulation 1,
`U = c_puct * P * sqrt(root.n)/(1+N) = 0` for every child (sqrt(0)), so all
children tie at score 0 and selection falls to the lowest-index legal move.
Sims 1–2 are therefore wasted on action 0 regardless of priors. This (a) wrecks
1–2 sim play and (b) — worse — biases the **low-sim self-play policy targets**
toward low-index moves, which is exactly the regime we now want to train in.

**Fix.** Treat the root's network evaluation as one visit: after expanding the
root set `root.n = 1` (and `root.w = root_value`). Then on simulation *k* the
numerator is `sqrt(k)` ≥ 1, so priors drive selection from the very first sim.
Child visit counts (used for the move/target) are unaffected by the offset.

**Result.** The cliff is gone; opt-rate is now monotonic in sims on the *same*
(100-sim-trained) net:

| eval sims | raw | 1 | 2 | 3 | 5 | 10 | 100 |
|-----------|-----|---|---|---|---|----|-----|
| before    | 0.950 | 0.587 | 0.587 | 0.945 | 0.973 | 0.984 | 0.992 |
| after     | 0.950 | 0.950 | 0.957 | **0.966** | 0.976 | 0.984 | 0.992 |

1 sim now equals the raw policy (a single sim just confirms the top prior), and
3-sim eval rises to 0.966. Confirmed the bug and the fix. Next: this also
de-biases low-sim self-play targets, so retrain *at* sims=3 and test both
constraints.

---

## Exp A — retrain at sims=3 with Change 1 (no other change)

Config: identical to the 100-sim run but `selfplay_sims=eval_sims=3`, full-set
3-sim eval each iter, 60 iters. Result:

- **peak opt(mcts) = 0.887, never reaches 0.90 → FAIL constraint 1.**
- But the curve is *flat and stable* the whole way (0.79 → ~0.88 by iter 8, then
  oscillates in [0.87, 0.887] through iter 60). No collapse, no drift. So the
  **stability mechanics are already fine**; the blocker is purely the *ceiling*.
- `opt(raw)` plateaus at ~0.85 (vs 0.95 for the 100-sim-trained net). The value
  head learns fine (`vloss` → 0.02). **Diagnosis: 3-sim visit-count policy
  targets are too weak** — with only 3 visits spread over the top-prior moves,
  the visit distribution carries little information, so the policy net can't get
  past ~0.85 raw, and 3-sim search lifts that only to ~0.887.

This is exactly the regime Gumbel AlphaZero targets.

---

## Change 2 — Gumbel root + completed-Q policy target

**Idea (Danihelka et al. 2022, "Policy improvement by planning with Gumbel").**
At low sim counts, replace two things:
1. **Acting:** select the self-play move by Gumbel-top-k sampling + Sequential
   Halving over a small set of considered actions, scoring by
   `logits + gumbel + sigma(Q)`. Guarantees a policy *improvement* over the prior
   even with 2-3 sims (unlike visit-count sampling, which is near-random then).
2. **Policy target:** instead of visit counts, use the *completed-Q* improved
   policy `pi' = softmax(logits + sigma(completedQ))`, where visited actions use
   their search Q and unvisited actions use a value-completion `v_mix`. This is a
   far stronger, lower-variance target at few sims.

`sigma(q) = (c_visit + max_b N_b) * c_scale * q` (paper defaults c_visit=50,
c_scale=1.0). Below the root we keep ordinary PUCT.

**Result (Exp B, sims=3, 60 iters).** Big jump over Exp A:

- **peak opt(mcts) = 0.954**, crosses 0.90 at **iter 3** (K=3).
- **Constraint 1: PASS** (0.954 ≥ 0.90 at 3 sims).
- **Constraint 2: PASS** — min over [K..3K=9] = 0.903; ran to 60.

So Gumbel at 3 sims clears both stated constraints. `opt(raw)` ~0.85 (the bare
net is still weak — Gumbel/search is doing the lifting, as intended).

**But a robustness flag:** the curve *peaks* ~0.954 at iters 15-30 then **drifts
down to ~0.92** by iter 60, with `vloss -> 0.000` (the value head overfits the
buffer; near-draw self-play makes z almost all 0). 3K=9 passes by the letter,
but the slow erosion means *very* long overtraining could eventually dip below
0.90. The next change targets that drift so the result is robust to arbitrary
overtraining, not just 3K.

---

## Change 3 — entropy regularization vs the overtraining drift

**Hypothesis.** The drift is policy over-sharpening: the completed-Q targets are
near one-hot (`sigma` with c_visit=50), the net keeps sharpening, self-play
diversity narrows, and the net overfits the buffer (vloss->0). An entropy bonus
on the policy loss (`loss -= entropy_coef * H(policy)`) keeps the policy from
collapsing and preserves exploration/generalization.

**Test.** Gumbel sims=3 + `entropy_coef`, run **120 iters** (≈40×K) to stress
overtraining far past 3K. Pass = stays ≥0.90 the whole way with no downward drift.

**Result (Exp C, Gumbel sims=3 + entropy_coef=0.02, 120 iters = 40×K).**

- peak opt(mcts) = **0.961**, crosses 0.90 at **iter 3**.
- **min over the ENTIRE 120-iter run = 0.908 — never dips below 0.90.**
- The curve does not drift to death: it oscillates in a bounded band
  [0.922, 0.961] and *recovers* to ~0.96 near the end. The mechanism is
  self-correcting: when the value head starts to overfit (`vloss -> ~0`) opt-rate
  dips, but the entropy bonus keeps the policy/exploration from collapsing and it
  bounces back. `opt(raw)` also rose (~0.85 -> ~0.90+).
- **Constraint 1: PASS.  Constraint 2: PASS** (robust to 40×K, not just 3×K).

Contrast with Exp B (no entropy): there opt-rate drifted *monotonically* 0.954 ->
0.915 over 60 iters and would likely breach 0.90 under longer training. The
entropy term converts that monotonic drift into bounded, self-correcting
oscillation — exactly the overtraining robustness the goal asks for.

---

## Summary

| run | search | sims | extra | peak opt | min over run | C1 | C2 |
|-----|--------|------|-------|----------|--------------|----|----|
| Exp A | PUCT (visit counts) | 3 | — | 0.887 | (never ≥0.9) | ❌ | — |
| Exp B | Gumbel (completed-Q) | 3 | — | 0.954 | 0.903 (drifting) | ✅ | ✅ 3K, drifts |
| Exp C | Gumbel (completed-Q) | 3 | entropy 0.02 | **0.961** | **0.908 over 40×K** | ✅ | ✅ robust |

**Recommended low-sim config:** Gumbel acting + completed-Q target,
`selfplay_sims = eval_sims = 3`, `max_considered = 8`, `c_visit = 50, c_scale = 1`,
`entropy_coef = 0.02`. Reaches ≥0.90 opt-rate at 3 sims by iter ~3, stabilizes
around ~0.95, and stays ≥0.90 through 40× the time-to-threshold with no collapse.

### What each change bought us
1. **Root-visit init fix** — removed the 1–2 sim degeneracy; necessary for any
   low-sim regime (it de-biases the policy targets).
2. **Gumbel + completed-Q target** — the decisive lever for *reaching* ≥0.90 at
   3 sims (0.887 → 0.954). Visit counts carry too little signal at 3 sims.
3. **Entropy regularization** — the decisive lever for *staying* ≥0.90 under
   heavy overtraining (monotonic drift → bounded oscillation).

### Open thread (not required by the goal)
The residual ~5% opt-rate gap and the mild oscillation both trace to self-play
covering a narrow (mostly-drawn) slice of the state space while opt-rate is
measured over *all* 4520 states. A coverage fix (starting a fraction of
self-play games from random openings) would likely push the floor above ~0.95
and shrink the oscillation further. Left as a future experiment.

---

## Change 4 — the "more eval sims, more losses to perfect" anomaly (root cause + fix)

**Anomaly.** Taking the 3-sim-trained net and only changing the *eval* sim count,
optimal-move rate rises monotonically (3→0.944 ... 64→0.991) but *losses to a
perfect opponent* get **worse**: mean losses/200 = 5.7 (3 sims) → 14.3 (32) →
30.0 (64), consistent across 6 opponent seeds. A more-optimal agent losing more
is paradoxical — investigated, not rationalized.

**Two hypotheses tested and the second confirmed (triangulated from two angles):**

1. *Final move-selection re-ranks pruned candidates* — **FALSE.** Switching the
   eval move to the most-visited action (the halving survivor) does **not** fix
   it (32 sims still ~16 losses) and lowers opt-rate. So the bug is in the search
   that builds the tree, not the final pick.

2. *The value head is queried out-of-distribution on a widening candidate set* —
   **TRUE.** Two independent probes converge:
   - **Width probe.** `m = min(max_considered, legal, max(2, n_sims))`
     (gumbel.py): at 3 sims the `max(2,n_sims)` term caps width at `m=3`; at ≥8
     sims it saturates at `max_considered=8`. So *raising eval sims silently
     widens the root candidate set 3→8.* 100% of the 83 on-path blunders at
     sims=32 are moves **outside the top-3 policy logits** whose **value-head Q
     over-rates them**. The net was trained with `selfplay_sims=3` → self-play
     only ever backs up the top-3 policy moves, so the value head is **OOD on
     low-policy moves**; wide-`m` eval queries it exactly there and search walks
     in. Decisive control: **sims=32 at width m=3 → 0 losses** (all 6 seeds);
     depth alone is purely beneficial, only width hurts.
   - **c_scale probe.** Down-weighting Q (`sigma`'s `c_scale` 1.0→0.1) also kills
     the spike (64 sims: 30.0→0.0 losses) — i.e. trusting the (good) policy prior
     over the (OOD) value also fixes it. Same mechanism from the other side.

   These decouple cleanly from global opt-rate: opt-rate averages over all 4520
   states (depth refines the bulk, so it rises); losses-to-perfect depend only on
   the few on-path positions vs a perfect opponent — exactly where wide-`m`
   exposes the value errors.

**Fix.** Keep the candidate **width identical at self-play and eval** by setting
`max_considered <= selfplay_sims` (here 3). Then eval sims is a pure depth knob.
Verified on the trained net:

| eval sims | width m=8 (buggy) | width m=3 (fix) |
|-----------|-------------------|-----------------|
| 3  | 5.7 | 5.7 |
| 16 | 3.5 | 4.3 |
| 32 | 14.3 | **0.0** |
| 64 | 30.0 | **0.0** |

(losses/200 vs perfect, mean of 6 seeds; opt-rate stays 0.98+). Applied as
`gumbel_max_considered=3` in `robust_lowsim_config`, `GumbelAgent` default
`max_considered=3`, and documented at the `m = ...` line in `gumbel.py`. Practical
upshot: **train cheap at 3 sims, then play near-perfectly by searching deeper at
the training width.**

The deeper, fully-general fix (if you ever want to eval at *wider* width than you
trained) is to train at that width so the value head is calibrated there — i.e.
width is a real hyperparameter that must match between data generation and play.

---

## Change 5 — does bleeding 10% of 10-sim (wider) games into training help?

**Idea.** Generate 90% of self-play at the cheap base (3 sims, width 3) and 10%
at 10 sims (width 8, since a 10-sim game naturally considers the wider set). The
hope: cheaply calibrate the value head on the wider candidate set so deep/wide
eval stops blundering. Cost: avg 3.7 vs 3.0 sims/game (~+23% self-play only).

**Method.** 3 training seeds each of baseline (mix=0) and mix (mix=0.1), 50 iters.
Eval each at the primary 3sim/w3 regime and the 32sim/w8 stress regime:
opt-rate on a 1000-state sample + losses to a perfect opponent over 450 games.

| regime | model | opt (mean) | vs-perfect losses/450 per seed | mean |
|--------|-------|-----------|-------------------------------|------|
| 3sim/w3 | baseline | 0.941 | [7, 56, 95] | 52.7 |
| 3sim/w3 | mix      | 0.951 | [103, 80, 27] | 70.0 |
| 32sim/w8 | baseline | 0.983 | [0, 34, 86] | 40.0 |
| 32sim/w8 | mix      | 0.983 | [93, 90, 0] | 61.0 |

**Verdict: inconclusive — dominated by training-seed variance.**
- The vs-perfect metric swings enormously across *training* seeds (baseline alone:
  7 → 95 losses). The baseline-vs-mix mean gap (52.7 vs 70.0) is far smaller than
  this spread, so it is not significant at 3 seeds (would need ~10+ to resolve).
- **Lesson:** a single run's vs-perfect number is unreliable at ~0.95 opt-rate —
  whether a model's residual ~5% blunders land on perfect-play lines is ~a coin
  flip per seed. opt-rate is the stable signal; vs-perfect is high-variance. (Our
  earlier single-run reads — "baseline clean", "mix catastrophic" — were both just
  draws from this wide distribution.)
- What *does* hold: the mix slightly *raises opt-rate* (0.941 → 0.951 at 3 sims,
  consistent in 2/3 seeds) for ~negligible cost. So richer deep-game targets help
  the stable metric a little; they do not reliably change perfect-opponent losses.

**Takeaway.** To actually cut perfect-opponent losses you must push opt-rate much
closer to 1.0 (state-space coverage + value calibration), not tune the sim mix.
The reliable lever for safe deep/wide *eval* remains Change 4 (match eval width to
training width). Keeping `mix_frac` available (default 0) as a cheap, slightly-
positive opt-rate nudge; not adopting it as a robustness fix.
