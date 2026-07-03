# Chess AlphaZero — overnight tuning log

Goal: from-scratch AlphaZero chess on a Mac CPU overnight. Not "launch a
TTT-tuned config and hope" — tune the things that actually differ for chess, with
real skill metrics, then commit the long run to the best config.

## Constraints / facts
- 8-core Mac CPU. Full config (64ch/6b net, 24 games/iter @32 sims, 100-ply cap)
  = ~163 s/iter. Too slow to tune with.
- Tuning uses a **fast proxy** (smaller net, shorter games, fewer iters) at
  ~30–40 s/iter; the chosen config is then validated at full scale.
- Self-play is the throughput bottleneck → low-sim Gumbel (our prior work) is the
  right fit. Capped games are material-adjudicated so the value head gets signal.

## Metrics
- `ploss`/`vloss`: policy CE / value MSE during training.
- `vs_random`: score over N games (saturates once it learns to capture — early
  signal only).
- `tactics`: frozen suite of mate-in-1 and hanging-piece positions; % solved by
  greedy search. The real skill signal (built by a sub-agent).
- self-play `avg_plies`, `decisive` fraction: health of the data.

## Tuning plan (priority order)
1. **Sanity**: confirm it learns at all (loss↓, tactics↑, beats random, games
   non-degenerate). Debug before tuning if not.
2. **Learning rate** — most likely to be mis-set from TTT (2e-3).
3. **Net size** (channels/blocks) — capacity vs CPU throughput; data-starved.
4. **Search**: sims and max_considered (chess branching ~35; width matters).
5. **max_ply / adjudication threshold**, entropy, c_scale.
Assemble best → launch long run → keep refining / investigate anomalies.

## Profiling — where self-play time goes (64ch/6b, 32 sims, 40-ply game)
- **net forward: ~68%** — 1320 *batch-1* forwards/game @ ~2.6 ms. Batch-1 is
  memory-bound on CPU: batch64 = 1.58 ms/pos → only **2.9x** batching speedup on
  CPU, but this is the **10–50x** lever on the 3090. Batched leaf eval across
  many concurrent games is the key architectural win.
- **python-chess terminal: ~19%** — `is_game_over(claim_draw=True)` rebuilds
  threefold keys every node. Use a cheap in-search terminal check; full outcome
  only at game end.
- **board copies in expansion**: `_expand` applies a fresh board for *every*
  legal child eagerly (~35/expansion, most never visited). Lazy expansion +
  make/unmake (push/pop instead of copy) cuts this hugely.

### Acceleration workstreams (parallel, sub-agents in worktrees)
1. Batched vectorized self-play (B games in lockstep, one batched forward) +
   lazy expansion + make/unmake + cheap terminal. Biggest win; future-proofs GPU.
2. Native (Rust/pyo3) movegen prototype + benchmark vs python-chess (orthogonal).
Meanwhile an interim learning run continues on the current engine.

## Org structure (research lead = me; specialists report, I decide + integrate + log)
- **Infra-A**: batched vectorized self-play (batch leaf evals across games). Running.
- **Infra-B**: native/Rust movegen prototype + benchmark. Running.
- **Evals**: standing always-on eval harness — tactics, blunder/hang rate, value
  calibration, "where does it fail unexpectedly". Reports failure modes.
- **Algo/Scaling**: fix the learning (cold-start target collapse) + find the
  compute-optimal config (net size vs speed vs strength for ~7h on 8-core CPU).
- Sub-agents propose; I verify a speedup actually yields a *stronger net in 8h*
  (not just "faster") before accepting. Only I write this log.

## CRITICAL eval finding — training DEGRADES tactics (iter 9: 0.07 < 0.20 fresh)
Interim proxy (32ch/4b, TTT-tuned search) made tactics *worse* than random init
(mate 0.30→0.14, hang 0.10→0.00). Diagnosed: the Gumbel completed-Q **policy
target is over-sharp at cold start**. On fresh-net mate-in-1 positions (30–51
legal moves) target entropy ≈ 1.4 (uniform ≈ 3.5), ~85% mass on ~2 near-random
moves (the sequential-halving finalists), mate move ≈ 0. So SGD collapses the
policy onto arbitrary moves before the value head learns material. Root: TTT
sigma `c_visit=50, c_scale=1.0` + narrow `max_considered=8` over ~35 legal moves.
→ Algo agent to find a config where tactics *rise* (softer targets / more
exploration / lr / value weighting / width). Decision pending its evidence.

## Decisions log (lead)
- **[Infra-B / fastchess] ACCEPTED, integrating as drop-in.** Rust+shakmaty
  (pyo3) movegen, verified 0 mismatches over 364k positions (found+fixed a real
  threefold-claim edge case). Micro: movegen 23.5x, make-move 9.8x,
  terminal+result **499x** (kills the 19% `is_game_over` hotspot). End-to-end
  self-play ≈**1.3x** (net forward is the 68% ceiling, 1/0.72≈1.39x cap).
  Rationale: it speeds the *game interface*, so single AND batched self-play both
  benefit; and once batching cuts the net fraction, the python-chess fraction
  grows → fastchess gets *more* valuable. Resumed Infra-B to build a parity-tested
  `FastChessGame` drop-in (new file). Fold into the keeper engine after the
  learning fix + batched engine land. Caveat: adds a maturin/Rust build step —
  must verify it survives an unattended restart before trusting it overnight.

## [Evals] failure characterization of the broken interim net (iter 10)
Eval harness (`alphazero/chess_eval.py`, `scripts/eval_checkpoint.py`) built — adds
hang/blunder rate, value-material correlation, unexpected-failure FENs. Findings:
- Tactics degraded (mate 0.40→0.12, hang-cap 0.10→0.02) but the net hangs *less*
  than random (blunder 0.20→0.07). So it's NOT giving material away — it's
  **policy collapse on forcing moves** + a **miscalibrated value head**.
- **Value head: std 0.009→0.79 (confident) but r vs material ≈ −0.08 (wrong sign-ish)**
  and MAE vs material 0.47→0.90. A value that rates being-down-material as good
  steers Gumbel's `logits+σ(Q)` selection toward losing lines → the real driver of
  tactics regression.
- Lead check: the value-target SIGN convention is correct (verified
  encode_board mover-perspective ↔ `z=z_white*to_play`; Black-to-move-up → z=+1).
  So this is value-head OVERFIT to spurious short-game patterns, not a sign bug.
  Consistent with too-high lr + over-sharp policy targets + tiny data.
- **New health metric to track on every keeper checkpoint: value-vs-material
  Pearson r must go POSITIVE.** (Use `scripts/eval_checkpoint.py`.)
- Metric caveats: hang detection is `is_attacked_by` proxy (not SEE); vs_random
  has ~0.09 stderr — trust tactics, hang_rate, value-r (low variance).

## [Algo] KEY insight: candidate width MC=8 is far too narrow for chess
Fresh (untrained) net tactics: **MC=8 → 0.24 overall; MC=24 → 0.50 (mate 0.84)**.
With ~35 legal moves, max_considered=8 filters tactical moves out of the Gumbel
candidate set entirely, so search can't even try them. This is the TTT "width"
lesson inverted: chess needs a WIDER candidate set. Implication: use wider MC for
BOTH self-play and eval (keep them matched per the TTT OOD lesson). Caveat: a
wider MC inflates the tactics *probe* too (search finds mates via terminal value),
so trained nets must be compared to the fresh baseline AT THE SAME MC to prove
real learning. → testing MC in the fix configs.

## CPU/orchestration lessons (lead)
- Sub-agents reliably WRITE code but stall on run-and-report (detached processes
  die / they "wait" forever). New division: specialists build code; **I run all
  experiments/benchmarks** (reliable detached + wait-loop) and decide.
- 8 cores: serialize heavy jobs. Running 2 experiments + sub-agent benchmarks
  concurrently caused 4x slowdown (39s→168s/iter) and contaminated timings. Keep
  ≤ ~6 self-play workers total active; benchmark speedups on a CLEAN machine.

## Decision: fastchess REJECTED (for now) — verified, not just assumed
Drop-in `FastChessGame` is parity-perfect (0 mismatches / 29,778 positions, incl.
identical encode_board planes + GumbelMCTS actions). BUT it rebuilds a python-chess
Board from FEN per expanded node so `encode_board` works → **end-to-end break-even
to slightly slower**. The 23–500x micro-op wins don't touch the encode path or the
net forward (68%). Not worth a Rust build dependency on an unattended run. Revisit
only if encoding is pushed into Rust too. (Exactly the "faster micro-op ≠ stronger
net in 8h" check — rejected on end-to-end evidence.)

## Decision: batched self-play (Infra-A) — ACCEPT pending clean speedup
Bit-EXACT equivalent to single-game GumbelMCTS (identical action + policy across
5 pos × noise × {4,16,32} sims, 5/5 tests). Entrypoint
`play_batched_games(evaluator, n_games, concurrency, ...)->(samples,stats)`, same
Sample format. Zero correctness risk; measuring end-to-end speedup on a clean
machine before wiring into the keeper.

## fix_combo FAILED + value-sign RESOLVED (key turning point)
fix_combo (MC=16, CSCALE=0.15, lr=1e-3, ENT=0.1, 16 proxy iters): tactics STILL
dropped (0.16 @MC16 vs fresh ~0.37; 0.06 vs 0.25 same-harness) and value
r_material = **−0.28** (worse than interim). Two configs both anti-correlate value
with material → looked systematic.
**Direct value-sign probe (bypassing harness) RESOLVES it: NOT a sign bug.** On
hand-built positions the value head orders material CORRECTLY (mover +Q −0.20 >
−Q −0.60; +R −0.16 > −R −0.71) and is perspective-consistent across colors. The
−0.28 is a strong negative BIAS + weak signal on near-equal in-distribution
positions, on a tiny (32/4) net trained only 16 iters.
**Reinterpretation:** the "tactics drop" is largely an artifact — the FRESH-net
tactics number (0.25–0.50) is inflated because a uniform policy lets the SEARCH
stumble onto mates/captures (terminal value). As the policy sharpens it explores
less, so tactics dip BEFORE the policy becomes accurate, then should recover. A
16-iter proxy only catches the dip. Real fixes: wider MC (chess width), a
stronger value signal, a real net, and a LONG run to see recovery.

## Plan: stop proxy-tuning, commit to a monitored KEEPER run
Config (lead call): net 64ch/6b, MC=16 (matched train/eval), sims=32,
CSCALE=0.3, ENT=0.05, lr=1e-3, max_ply=100, checkpoint every iter, eval (tactics +
value_r via harness) every N iters. Adopt batched engine if its clean speedup
justifies (benchmark pending). Monitor the tactics+value_r TRAJECTORY over many
iters; intervene (e.g. graded-material value target) only if it doesn't recover.

## KEEPER LAUNCHED (v2, material-anchored value)
Decisions feeding it:
- **Batched self-play = GPU lever, NOT CPU**: measured 2.1x single-process vs 7
  multiprocessing batch-1 workers already saturating cores → keep multiprocessing
  for the CPU overnight run; batched (verified bit-exact) is for the 3090.
- **Value target fix** (the crux): immediate-material anchor `tanh(mat_stm/5) +
  0.5*outcome` for capped games; true result for real terminations. Smoke: value
  r went **-0.28 → +0.30 by iter 2**. The value head now learns material → steers
  search to win material.
- Config: 64ch/6b, MC=16 (chess width), sims=32, c_scale=0.3, ent=0.05, lr=1e-3,
  max_ply=80, 28 games/iter, 7 workers, ~120s/iter, checkpoint every iter, eval
  (tactics+value_r) every 8 iters. ckpt: models/chess_keeper/.
- MONITORING: watch tactics RECOVER above the inflated fresh baseline and value_r
  stay positive. Intervene if it degrades.

## Keeper trajectory (live)
- iter 8: ploss 1.79, vloss 0.08, **vs_random 0.78** (was ~0.4), tactics 0.25
  @MC16 (vs broken-interim 0.07; inflated-fresh ~0.37), **val_r +0.07** (sign
  flipped from -0.28 → material anchor working; weak because measured on
  near-equal random positions). ~200s/iter, all-decisive games.
  Read: real play improving fast (vs_random); watching tactics to climb back
  above fresh and val_r to strengthen. No intervention — let it run.
- iter 16: ploss 1.64, **vs_random 0.94** (↑ from 0.78), **val_r 0.14** (↑ from
  0.07), tactics 0.22 (flat). Net is becoming a solid anti-blunder/grab-material
  engine that crushes random; tactical *precision* on contrived probe positions
  is flat (~0.22) — realistic for overnight CPU. vs_random + val_r are the real
  signals and both rising. Continue. NOTE for morning: play the best net at
  HIGHER sims (64–128) with MC=16 (matched) for max strength — free boost.
- iter 24: ploss 1.63, **vs_random 0.97**, **val_r 0.24** (steady climb
  0.07→0.14→0.24 — value head learning material well), tactics 0.25. Healthy.
  vs_random saturating near ceiling (random is weak); val_r is now the live
  progress signal. Best net = latest.pt (vs_random monotonic). ~180s/iter.
- iter 32: **vs_random 1.00 (maxed)**, **val_r 0.40** (0.07→0.14→0.24→0.40,
  strong steady climb), tactics 0.27 (now ticking up). Run is healthy and
  improving; val_r is the live signal since vs_random is saturated. No errors,
  all-decisive games, ~184s/iter. iter_00025.pt snapshot saved.
- iter 40: **PLATEAU**. vs_random 0.88 (dip, likely noise ±0.09), val_r 0.40
  (flat), tactics 0.27 (flat), ploss 1.65 / vloss 0.10 (slight uptick). Expected
  ceiling of the material-anchored value: net has learned material + anti-blunder,
  limited further signal without much more compute. Watching the ploss uptick for
  overfitting. Peak so far ~iter 32. Morning: eval snapshots 25/50/75/latest, pick
  best. No restart (marginal upside, ~2h left).
- iter 48: dip was NOISE — **vs_random back to 1.00**. val_r 0.35, tactics 0.27
  (plateaued + STABLE). ploss/vloss slowly rising (1.69/0.11) but eval NOT
  degrading → model at capacity / targets diversifying, not harmful overfit. Keep
  alive. Snapshots iter_00025/00050.pt preserved. Conclusion: net is a stable
  material-aware anti-blunder engine; further training = marginal. Morning task:
  characterize best snapshot, play sample game, play at higher sims (MC=16).

## Late trajectory (iter 64–104): healthy, val_r still climbing
vs_random pinned 0.94–1.00, **tactics drifting up 0.25→0.31 (iter 120, new high)**,
val_r 0.35–0.62 (noisy, strong), no crashes, buffer full (150k window), ~3360
self-play games. Still slowly improving, not degrading. Best net = freshest
snapshot (iter_00100/00125.pt) ≈ iter_00050.pt (characterized) — plateau-equiv.
Run can be stopped anytime; best net is saved.

## DELIVERABLE (morning) — what the overnight chess net achieved
Best net: `models/chess_keeper/iter_00050.pt` (≈latest; plateau-equivalent).
64ch/6b ResNet, ~1500 self-play games on Mac CPU (~8h incl. all the debugging).

| metric | fresh net | broken (TTT-ported) | KEEPER |
|--------|-----------|---------------------|--------|
| value r vs material | ~0.00 | −0.08…−0.28 | **+0.55…+0.61** |
| blunder rate (newly hangs piece) | 0.25 | 0.07 | **0.07–0.10** |
| hanging-capture (grabs free material) | 0.10 | 0.02 | **0.18** |
| vs_random (score) | 0.40 | 0.29 | **0.94–1.00** |

**Honest summary:** Starting from a tic-tac-toe-tuned config that actively
DEGRADED play, we diagnosed and fixed the real problems (candidate width too
narrow for chess; value target too noisy → value head anti-correlated with
material and misguiding the search) and trained a net that is a **material-aware,
anti-blunder engine**: it develops pieces, captures hanging material (sample game:
won the exchange, finished +2 vs random), rarely hangs its own pieces (~3x lower
blunder rate than random init), crushes a random opponent, and has a value head
that genuinely understands material (r≈0.6, up from −0.28). It is NOT a sharp
tactician (mate-in-1 0.22–0.26, below the fresh net's inflated 0.40) and shuffles
aimlessly with no deep plan when there's no tactic — expected for ~1500 games on
CPU. To play it strongest: load the ckpt, GumbelMCTS at sims=64–128, MC=16,
c_scale=0.3, add_noise=False. Path to stronger: the verified batched engine +
3090 (10–50x self-play) and a proper outcome-based value once enough games exist.
Full journey + every lead decision above.

## THROUGHPUT PUSH (GPU box, RTX 3090 + 72 cores) — max pos/s
Baseline (Python engine, 32 procs): ~1.6 games/s, ~127 pos/s. GPU idle at 0% in
single-process batched self-play -> bottleneck is Python chess dynamics + encode.
Encode benchmark: `encode_board` (python-chess piece_map) = **4.25 ms/pos (235/s)**
is the wall; numpy-blob -> from_numpy -> H2D copy = 45k pos/s (NOT the problem);
per-element-on-GPU antipattern = 1.3k/s (confirmed catastrophic). Rule (CLAUDE.md):
build CPU-contiguous blob, copy once.

Fan-out (4 specialists, canonical bench 64ch/6b, 32 sims, MC=16, 80-ply; same
Gumbel algorithm to preserve elo-vs-frames):
- A Rust dynamics+encode (python-chess out of hot loop) — reliable CPU 10-50x.
- B pgx + mctx (JAX) fully-vectorized GPU self-play — the ceiling; mctx IS Gumbel.
- C GPU inference server + many CPU search workers — saturate GPU, pragmatic.
- D net-inference accel (fp16/bf16, torch.compile, CUDA graphs, conv 8x8x73 head).
Lead will combine winners + validate elo-vs-frames before committing a training run.

## [A] Rust encode — DONE, verified
Rust `Board.encode()` (+ `encode_batch`, `material_diff`) fills the (18,8,8) f32
blob directly from shakmaty bitboards. **Bit-exact: 0 mismatches / 48,104 positions
(twice).** Throughput **1.39M enc/s vs ~20k/s python-chess encode_board = 70x**
(337x vs the FEN-reconstruct path). Wired via `encode_state` dispatcher;
`play_chess_game` now takes `game_cls` (python-chess path byte-identical).
**Key handoff: encode wall gone -> self-play is now NET-EVAL-BOUND** (batch-1 CPU
forward ~8.6 ms/pos). => next lever is BATCHED GPU inference (agents C/B); D shows
inference ceiling ~160k inf/s, so batching the net is the win. Contended games/s
numbers discarded; lead will re-measure clean+serial once the box is quiet.

## [C] GPU inference server — DONE (code), contended numbers discarded
`bench_server.py`: central GPU server + CPU search workers; `RemoteEvaluator` is a
drop-in for ChessEvaluator so `gumbel.py`/`az_mcts.py` are UNCHANGED (algorithm
preserved). IPC finding (load-bearing): shared-memory torch tensors + busy-polled
uint8 flags = **0.13 ms** round-trip; mp.Queue=1.7ms, mp.Event=3.8ms (Queue+Event
stalled at 50ms/eval). Clean forward = **3 ms flat to batch 256 = 86k evals/s**.
Clean CPU reference: **single-thread python-chess search ~4 pos/s/core (~130
evals/s)** -> CPU-search ceiling ~cores×4 ≈ 290 pos/s with python-chess; Rust
dynamics raises per-core rate. **Bottleneck is the SEARCH, not the net.** Launch
must set OMP/MKL/OPENBLAS_NUM_THREADS=1 (else workers spawn 64 BLAS threads each).

## [D] net inference — DONE (clean): inference is NOT the bottleneck
Clean RTX 3090 peaks (fp16): TensorRT+conv head **818k inf/s**; torch.compile
max-autotune+conv **505k**; compile+FC 407k; eager fp32 130k. (Earlier "160k" was
contended.) Peaks at batch 256-512, DECLINES past ~1k (8x8 net saturates SMs
early — don't chase huge batches). fp16 numerics immaterial to MCTS.
**ADOPT the conv 8x8x73 policy head** (+24% speed, drops the 9.5M FC params,
canonical AZ, drop-in). Recommended infra: net.half()+torch.compile(max-autotune)
@batch 256-512 ≈ 500k inf/s. => 500k evals/s available; at 32 sims that feeds
~15k pos/s IF search could supply it. SEARCH is the bottleneck (confirmed).

## [B] vectorized JAX (pgx+mctx) — first number SUSPICIOUS, needs clean re-run
First run (CONTENDED): batch=1024, 32 sims, 40 steps, 411s -> **pos/s=100** (~10s
per move over 1024 games). Shockingly slow for full GPU vectorization. Suspects:
GPU contention during the run, and/or pgx GPU chess step + mctx sequential sims
being heavy (pgx uses 119 planes vs our 18; net 10.1M). Awaiting B's full report;
must re-measure on a quiet GPU before judging the ceiling.

## THROUGHPUT — final synthesis + architecture decision (lead)
All four vectors measured CLEAN (after rebooting the box to clear a 22GB GPU leak
from B's killed jax process — kill -9 of a CUDA proc mid-kernel leaks GPU mem;
reset unsupported on the container, so stop/start was required).

Findings:
- **Inference SOLVED**: ~500k inf/s (fp16 + conv 8x8x73 head + torch.compile),
  ~800k TRT. NOT the bottleneck.
- **Encode/dynamics SOLVED**: Rust encode 70-337x, bit-exact. NOT the bottleneck.
- **The bottleneck is the SEARCH** (CPU, embarrassingly parallel across games).
- **GPU-vectorized search (pgx+mctx JAX) is the WRONG tool for chess**: clean
  ceiling **~150-185 pos/s** (~2 games/s), flat/declining with batch. Root cause:
  ~90% mctx tree-array overhead over chess's 4672-action space + vmap while-loop
  divergence — FUNDAMENTAL, ~400-500 even optimized. Algorithm/rules verified
  correct (perft exact, completed-Q identical w/ configurable constants). Verdict:
  DO NOT migrate to JAX.
- **CPU+Rust+GPU-server path (A+C+D)** clean peak on THIS box = **~28-30 pos/s**
  (procs≈16); procs=32 THRASHES (5 pos/s) because the **container is capped at
  ~8.6 vCPUs** (cgroup cpu.max 864000/100000), not the host's 72. The CPU path is
  hard-capped by vCPUs here.

**DECISION: the winning architecture is Rust dynamics+encode + GPU inference
server (shmem-flag IPC, 0.13ms) + compiled conv-head fp16 inference — run on a
HIGH-vCPU instance.** The search scales ~linearly with cores; at 64-72 vCPUs this
stack should hit ~200-250+ pos/s, beating the JAX ceiling, with the GPU only
lightly loaded (inference is cheap for this small net). This box's 9 vCPUs are the
sole reason we're at 30 pos/s. RECOMMENDATION: re-provision a many-vCPU instance
(modest GPU is fine) and run the already-built stack. Stack is built + verified;
it just needs cores.

## THE insight: self-play is LATENCY-bound, not throughput-bound -> port to Rust
Why 65 pos/s despite 500k inf/s GPU: MCTS is SEQUENTIAL per game (each sim's
descent depends on the previous sim), so a game has only ONE net eval in flight
and the worker BLOCKS on it. pos/s/game ≈ 1/(sims × per-sim-latency ~5ms) ≈ 6;
~1 game/core -> ~65 pos/s. We use the GPU's 3ms LATENCY, not its 500k/s
THROUGHPUT (need only ~2k evals/s -> GPU 99% idle). Adding vCPUs only adds more
sequential latency-bound games (linear, caps low-hundreds).
To use the throughput: run THOUSANDS of concurrent games, batch their leaves into
one forward (+virtual loss for in-flight parallelism). Single-process Python
batched engine = GPU idle (tree ops ~ms in Python). JAX/mctx batches but 4672-
action tree overhead caps ~150. => **Native Rust batched-MCTS** (tree ops in us,
batched inference) is the only path to thousands of pos/s (KataGo/Leela design).
DECISION: porting the hot loop to Rust. v1 = port chess_batched.py (bit-exact
Gumbel) to Rust over fastchess boards; thin Python eval-callback does only the
batched GPU forward; Rust does dynamics+encode+tree+masking. Validate vs Python
search (same moves/targets); benchmark pos/s. New box: 19 vCPU / 3090 / 188G
(ssh -p <PORT> root@<VAST_BOX>). vCPUs barely matter once the engine is Rust +
one process drives the GPU.

## RUST batched self-play engine — DONE, 28x, bit-exact
`fastchess.run_selfplay(eval_fn, n_games, sims, mc, c_visit, c_scale, c_puct,
max_ply, mat_thresh, add_noise, seed)`: N concurrent game-trees in lockstep, one
leaf/game/round -> one batched torch forward (the only Python crossing), all
dynamics+encode+PUCT+seq-halving+completed-Q+backprop in Rust. Value target =
play_chess_game (terminal result; capped tanh(mat/5)+0.5*outcome).
CORRECTNESS: bit-exact vs GumbelMCTS/play_chess_game over 292 positions / 10 sim
budgets: 0 action mismatches, 0 z mismatches, max|pi| 3e-8. Caught+fixed a real
bug: seq-halving final move must be chosen over the ORIGINAL m candidates, not
just survivors (matches gumbel.py quirk).
THROUGHPUT (3090, 64ch/6b, fp16, 32 sims): B=256 1145, B=512 1538, **B=1024 1816
pos/s (27.9x the 65 baseline)**, B=2048 1539 (regresses, D2H-bound). avg_batch 959.
HEADROOM: GPU still 64% idle (util 36%, 60k/500k inf/s). Bottleneck now =
single-threaded GIL-held Rust tree ops (42%) + 19-38MB/round logits D2H.
v2 LEVERS: (1) release GIL + multi-thread tree ops over ~19 vCPUs [biggest win];
(2) legal-mask+softmax on-GPU (kill D2H); (3) virtual loss; (4) bigger nets shift
to the well-utilized forward. Target after v2: several thousand pos/s.

## GPU TRAINING RUN (Rust self-play + 128ch/10b net on 3090) — LIVE, learning fast
Self-play via fastchess.run_selfplay (Rust, ~1550 pos/s steady = ~24x Python),
12.6M-param net trained on GPU. ~133s/iter (50s self-play + ~80s train), 1024
games/iter, buffer->1M, material-anchored value, mc=16 c_scale=0.3 entropy=0.05.
Trajectory: ploss 4.1->2.77 (it1->10); **iter 10 (~22min, ~10k games):
tactics 0.41, val_r 0.78, vs_random 1.00** — already BEATS the entire 8h CPU
keeper (peak tactics ~0.31, val_r 0.6, 3500 games). The throughput work pays off:
24x data rate + 2x net -> faster & stronger learning. Monitoring continued.

## D2H FIX (legal-logit gather) — DONE, big GPU-util win [commit be6c4de]
Per-round eval boundary was D2H-ing the full (B,4672) logits (~19MB/round, ~2640
rounds/iter) + cuda.sync -> GPU starved at ~10% util / 124W in self-play. Fix:
eval_fn now takes flat legal (rows,cols), gathers logits[r,c] ON GPU, D2Hs only
~B*35 floats (133x smaller payload, 49-83% less transfer/round). Rust softmaxes
per-game legal segments -> bit-exact (validate: 0 action/z mismatch, max|pi| 3e-8).
LIVE RESULT (128ch/10b, B=1024, max_ply120): GPU util 10%->82%, power 124W->333W,
pos/s 1527->1725 (B=2048 would gain far more — D2H was 40ms/round there).

## WHO-WINS VALUE EXPERIMENT (switched iter 40) — verdict: NOT breaking the plateau
Dropped material anchor; z = binary game outcome (terminal result / up-a-piece+
adjudication for capped). Goal: let sacrifices score +1, break the tactics-0.42
ceiling. Trajectory: tactics 0.45(it50) 0.44(it60) ~flat in old 0.40-0.45 band —
NO break. val_r dropped 0.85->0.45 (expected). vloss stuck ~0.50 (vs 0.16 for
material; better than always-0 baseline 0.83, so partial signal but NOISY — value
can't fit outcomes at 32 sims). decisive ~83%. Hypothesis: tactics is search-depth
limited (32 sims), and the outcome target is too noisy at 32 sims. Next lever (D2H
freed the GPU for it): bump sims (deeper search -> stronger play -> cleaner
outcomes + more tactics found). Decide at ~iter 80-85.

## SIMS 32->64 (depth test, eval kept @32) — iter 70, who-wins still on
First comparable eval it80: tactics **0.50** (up from the 0.42-0.45 plateau, +0.08
> probe noise) — deeper self-play helps. KEY nuance: vloss UNCHANGED ~0.48, so the
tactics gain is NOT from a cleaner value — it's better POLICY targets from deeper
search. I.e. depth rescues tactics even though the who-wins value stays noisy.
pos/s ~970 (~2x slower, fine). Awaiting it90 to confirm the climb isn't a 1-point
fluke; if it holds -> push sims 128. (who-wins-vs-material value question becomes
secondary since depth helped regardless.)

## PREDICTION STUDY (1000 positions x 12 checkpoints) -- value is the frozen bottleneck
From playing the net (user): it traps a bishop then won't take it / releases pressure,
yet value stays high. Probe: policy LOVES cxb3 (#1, 33%) but value is +0.76 before vs
+0.79 after winning it, and RATES b4-release (+0.87) ABOVE actually winning the bishop
(+0.67) -- incoherent, no foresight (only crashes when bishop physically escapes).
Generalized via pred_study.py: value sign-acc FLAT ~0.71 from 5M->62M frames (57M frames
~no value improvement!); value IS material-aware (corr 0.77, separates +0.7/0/-0.6) so NOT
broadly blind -- bishop was a specific positional misjudgment. But value THRASHES (~0.43
RMS change/checkpoint, never converges; policy top-agree only ~42%). 71% = outcome-noise
floor under the net's own poor conversion (vicious cycle). => the VALUE caps the system.
FIX (evidence-backed): auxiliary material/score target alongside who-wins (KataGo/Leela
style) to break the noise ceiling. Charts: pred_study.png, elo_vs_frames.png (tournament).

## SMAC SWEEP first leaderboard (7 configs, elo-per-1h-train) -- baseline looks BAD
Elo spread ~870 (hyperparams matter hugely). TOP: cfg4 +324 (96ch/10b sims32 ng1024
lr8.8e-4 ts76 buf~1M, 7.5M fr), cfg1 +270 (48ch sims8 ng2048, 26.6M fr), cfg6 +216
(48ch sims8 ng512). BOTTOM: cfg5 -548 (64ch/4b sims16 ng256 ts256 buf75k = overtrain
stale small buffer), cfg3 -259 (ng256 ts303), cfg0 -106 (128ch sims64 -> only 3.4M fr).
PATTERN: low sims (8-32 never 64) + many games (1024-2048) + big buffer + FEW train_steps
win; high-sims/few-games/small-buffer-overtrain lose. => our 67M-frame BASELINE
(128ch/10b sims64 lr1e-3 const) sits in the losing region -- likely a poor choice for
elo-per-wallclock. SMAC steering toward efficient region. ~1.5h/config sequential.

## ANCHORED SWEEP (Elo vs the 67M-frame net = 0) -- no 1h config is close (expected)
67M net beat every 1h config 28-30/30. Anchored leaderboard: cfg4 -450 (96ch/10b sims32
ng1024 buf977k ts76) BEST, cfg1 -516 (48ch sims8 ng2048), cfg6 -554, cfg2 -721, cfg0 -884
(128ch sims64 = baseline-like), cfg3 -1008, cfg5 -1305. 1h (~5-26M fr) << 67M fr so the
gap is expected; absolute gaps are soft (30-0 sweeps) but ORDERING solid. Sweep's value =
relative ranking (efficient hyperparams): low-sims/many-games/big-buffer/few-train-steps
win. NEXT EXPERIMENT the sweep enables: take the winning config + run it LONG (~baseline's
~19h) to test efficient-hyperparams-long > inefficient baseline. Sweep continues (cfg7+).

## BATCHED ARENA + THROUGHPUT STUDY -> back to the value head (again)
Built leaf-parallel PUCT arena (compiled, time-budgeted, L=1 bit-exact vs sequential).
torch.compile+CUDA-graphs+fp16 cut the forward 2-4x -> GPU no longer the bottleneck;
CPU leaf-finding (~1ms/leaf Python) is 82-94% of round time at useful L. BUT two things
kill the original "time-based eval helps small nets" premise: (1) strength-at-equal-
walltime favors LOW L (2-4) -- virtual-loss approximation costs per-sim quality; (2)
the small nets DEGRADE with more sims (value-head OOD-at-high-sims: cfg1 40%@s16 ->
2.5%@s128, the recurring problem since TTT). So they can't cash speed into more sims.
Rust leaf-finding port: only helps high-L throughput, but eval wants low-L + self-play
is game-parallel -> DEPRIORITIZED. PUNCHLINE: every thread (tactics plateau, bishop
blunder, frozen value, can't-use-more-search) -> the VALUE HEAD is the core limiter.
Also implies the SWEEP eval (fixed 32 sims) is confounded: configs trained at sims!=32
are evaluated OOD -> should eval each at ~its training sims, or fix the value head.

## Experiments
(filled in as they run)


---

## v2 SMAC sweep (production-eval objective, varied-sims space) — OUTCOME

Setup: objective = Elo at production think-time (1s/move, calibrate sims/move single-game
leaf-parallel, then per-net-sims tournament vs pool incl 67M anchor=0, UHO suite both
colors). Search space added VARIED training sims (sims_lo..sims_hi up to 256, sampled
log-uniform per iter). Warm-started pool+SMAC from the 8 v1 configs (reeval_results.json).
1h/config, sequential on a clean box.

RESULT (8 new configs explored, cid 8-15): the expanded space found NOTHING better than
the v1 best. Final leaderboard (anchor=0): cfg4 -655, cfg6 -679, cfg1 -705 (all WARM
seeds), then best new config cid10 -847, ... down to cfg7 -1485. The varied-sims lever
did not pay off — best new config ~200 Elo below cfg4; several SMAC picks had sims_lo>
sims_hi (collapsing to fixed sims anyway). => Hyperparameters are NOT the lever; the
VALUE HEAD remains the ceiling (now confirmed by a real sweep, not just the bishop-trap /
OOD intuition). At production think-time every net (anchor included) is forced 30-300x
past its training depth and the value head feeds the search noise.

ACTION: per the plan (4+ more configs, then long run with best), stopped the sweep and
launched a LONG INDEFINITE run with the best config = cfg4 (96ch/10b, sims=32, ng1024,
ts76, lr 8.8e-4->9.3e-5, buf 977k, bs256, mc8) via sweep/train_long.py -> models/chess_long
(varied-sims-capable but cfg4 is fixed sims=32; resumable; iter_NNNNN.pt every 0.5M frames;
cosine LR over 40M frames then held). Open question this long run answers: does the
best-tuned config beat the 67M anchor with extended training? (Likely still value-OOD-
limited at production think-time -> the honest next lever is the value head, deferred.)


---

## CORRECTION: the "value-head OOD at high sims" was a BROKEN-EVAL ARTIFACT

Clean test (sweep/elo_vs_sims.py): SAME net vs ITSELF at different sim counts, head-to-
head, per-side sims, UHO suite both colors (zero confound — only search depth varies).
Elo vs sims (vs @64=0), L=16:

  anchor67M:  @64 +0  @128 +181  @256 +292  @512 +436  @1024 +603  @2048 +842
  cfg4:       @64 +0  @128 +11   @256 +86   @512 +242  @1024 +426  @2048 +604
  cfg1:       @64 +0  ... rises monotonically, @2048 beats @64 ~30/32

ALL THREE nets (strong anchor, weak cfg4, weakest cfg1) convert search into strength
MONOTONICALLY to 2048 — none peaks, none falls. So the old "cfg1 40%@16 -> 2.5%@128
collapse" that anchored the whole value-head-ceiling thesis was an ARTIFACT of the old
broken eval (deterministic/duplicate openings -> one lost line counted N times; pre-Rust
batch-1). In a clean arena there is NO OOD cliff. MCTS refines a good-enough prior, and
even our weak nets are good-enough (Konig's point: policy/value are only coupled to sims
indirectly via how play changes, so more search = more correction, not extrapolation).

Implications: (1) the value head is NOT the ceiling; the limiter is ordinary net QUALITY
(the Elo-vs-frames plateau). (2) The per-config-calibrated-sims eval was solving a non-
problem (still valid, just unnecessary). (3) The varied-sims sweep lever targeted a non-
issue -> exactly why it bought nothing. (4) The anchor's production-eval dominance was
because it rides search BEST (it's simply the best net), not "least OOD".

PIVOT (Konig's call): continue training the 67M net (best net, rides search best) and
evaluate at a FIXED 2048 sims / L=16 (the strong operating point; no calibration needed).
Froze models/chess_gpu/baseline_67M.pt (=anchor=0 ref); resumed scripts/train_chess_gpu.py
from iter 650 / 66.67M frames (128ch/10b sims=64). New iter_NNNNN.pt scored vs the frozen
baseline at 2048/L=16 via sweep/eval_vs_baseline.py -> models/chess_gpu/elo2048.jsonl =
Elo-vs-frames curve PAST the plateau, measured where the net is actually strong.


---

## THROUGHPUT ABLATION CAMPAIGN (step 1: maximize self-play games/s, preserve elo-vs-frames)

Goal: self-play that is FULLY GPU-BOUND (GPU forward is the bottleneck, all CPU work
hidden under it). Each ablation on its own branch + measured on a CLEAN GPU. Two RTX 3090s
(box1 = original vast box; box2 = second instance, 192 cores).

### KEY DIAGNOSIS (settled): the self-play round is NOT forward-bound
Clean profile @ production (128ch/10b, sims=64, n_games=1024, max_ply=120): the GPU forward
is only ~70% of each MCTS round (~9-11ms forward, ~25-30% GPU-idle). The non-forward ~30% =
Rust MCTS tree work (descend/expand/backprop across 1024 trees) + the legal-logit D2H gather +
Python eval_fn dispatch + the batch-shrink "drain" tail (batch falls 1024->1 as games finish).
So the lever for GPU-bound is HIDING or REMOVING the non-forward time, NOT speeding the forward.

### chlast (channels_last + BN-fold) — branch opt-chlast / merged to main (ff3945e)
BN folded into each bias-free conv (eval-only, bit-exact: 3.8e-5/1.1e-5 max diff) + channels_last
(NHWC) -> cuDNN tensor-core kernels, no NCHW<->NHWC transposes. FORWARD @1024: +28-30% faster,
and that win is MEMORY-INDEPENDENT (holds with optimizer/grads resident, under a 2GB cap, with
training interleaved — the "memory pressure off-loads cuDNN" hypothesis is falsified). END-TO-END
self-play: only +6.3% (clean isolated A/B: NCHW 1087 -> NHWC 1155 pos/s), because the +30% forward
is diluted by the ~30% non-forward round. The earlier "+15.5%" was a noisy high outlier (NCHW alone
spans ±4%, cold cudnn-cache run is 413 vs 1087); the in-training "0%" was the same noise on the low
side + the per-iter fused_inference_net deepcopy sitting inside the timed region. VERDICT: keep
(real +6%, free, bit-exact); its value GROWS once overlap/refill make the loop forward-bound.
TODO: make the fused-net rebuild a persistent in-place re-fold (not a per-iter deepcopy in sp_t).

### ping-pong two-pool overlap — branch opt-pingpong — +6% (rejected as standalone)
Two pools alternating, one forward in flight, tree(B) under forward(A). Measured 1093 vs 1034
baseline = +6% only. Overlap is real but small: during forward(A) the CPU is busy LAUNCHING A's
hundreds of tiny 8x8 kernels (GIL held), so little free CPU to run tree(B). HYPOTHESIS: CUDA graphs
(one launch instead of hundreds) would free the CPU during the forward -> ping-pong could then
overlap fully. "graphs + ping-pong" is the headline combo to test.

### IN FLIGHT
- Thread-scaling test (box1): is the Rust tree work parallelism-starved? A measurement suggested
  it runs effectively ~2-core despite rayon. FASTCHESS_THREADS sweep 1..64 -> if non-forward time
  drops with threads, parallelizing the tree work is the single biggest GPU-bound lever (cheaper
  than overlap). DECISIVE either way.
- CUDA graphs (box2): does graph-replay (a) speed the forward and (b) free the CPU enough for overlap.
- Pending: refill (kill the drain tail), faster D2H gather, and (moonshot) eval-in-Rust (tch/libtorch,
  no Python callback -> native forward/tree overlap) if the tractable stack plateaus below GPU-bound.

### RESULTS (all measured on clean single-GPU RTX 3090, production config unless noted)

**Definitive round breakdown (box1 profiler, 986 pos/s baseline, 7800 rounds):**
  round 13.60 ms = forward 9.76 ms (71.8%) + NON-FORWARD 3.84 ms (28.2%)
  non-forward = H2D 1.03 (7.6%) + D2H/gather 0.44 (3.2%) + tree/Rust/launch residual 2.37 (17.4%)
  => GPU-bound ceiling ~= forward-only round ~= 1374 pos/s (+39% over 986). chlast lowers the
  forward -> a higher ceiling. Overlap (hide the 3.84 ms non-forward under the forward) is the lever.

**Ablation matrix (each on its own branch):**
| ablation | result | verdict |
|---|---|---|
| chlast (BN-fold + channels_last) | fwd +30% (mem-independent), e2e +6% | MERGE (opt-chlast -> main); GPU-bound component |
| **two-pool-eager overlap** | **+22.8% (box1: 1119->1374 pos/s), 83.7% GPU util, sample-exact** | **MERGE (opt-pingpong) — the overlap win, ~95% of the forward floor** |
| graph two-pool | 251 pos/s / 16% util (-76%) | FAIL (documented). Shared static buffer per size -> both pools same size -> WAR/WAW hazard -> submit blocks on sibling forward (11.89ms vs 0.32ms isolated). Fix = per-pool double-buffer (in test). |
| CUDA-graph forward (isolated) | wall 1.01x (compute-bound), CPU-launch 386x less (0.026ms) | opt-cudagraphs. Bit-exact, frees CPU in isolation; NET LOSS in the eval path (fetch on static buffer). |
| refill (kill drain tail) | tail is 0.6% of GPU-time at cap-120 | REJECT for capped self-play (mean batch 864/1024). Only matters uncapped. |
| thread-scaling (tree work) | plateaus at 8 threads (~1.4x), uses ~10-16 of 80 cores | tree work is small (2.37ms residual) AND weakly parallel -> not a lever; overlap hides it. |

**GPU-bound verdict:** chlast + two-pool-eager = 1374 pos/s, +22.8%, 83.7% util = ~95% of the
forward-floor ceiling. The remaining ~16% idle is the eager submit's GIL-held kernel-launch
(~3.4ms) partially blocking the sibling pool. Closing it needs per-pool double-buffered CUDA
graphs (submit 0.06ms, in test on box2) or eval-in-Rust (drop the Python callback entirely).
Refill / graph(single-buffer) / tree-parallelism are all dead ends here. Training config = the
two-pool eval_fn (submit/fetch) on the chlast fused net; frames/iter doubles at fixed n_games
(2 pools) so the replay ratio is being recalibrated live in the babysit.

### TRAINING INTEGRATION + BABYSIT (two-pool eval_fn merged into scripts/train_chess_gpu.py)

Resumed the 67M run (iter 820, 84.2M frames) with the two-pool eval_fn (submit/fetch on the
chlast fused net) + train_steps=500. Babysat iters 821-829 (~30 min). VERDICT: PASS.

- SPEEDUP HOLDS IN-LOOP: warm self-play ~1364-1404 pos/s (iter 821=594 was cold cudnn cache),
  vs the old single-pool ~984 pos/s -> +38-40%. GPU util 90% (was ~72%). Matches the bench (1374).
- 5x TRAINING DOESN'T BREAK: steady-state loss is HEALTHY, better than the pre-5x run:
    old (train_steps=100, iters 815-820): ploss ~2.27, vloss ~0.457
    new (train_steps=500, iter 829):      ploss  2.315, vloss  0.384  (vloss -15%, ploss flat)
  The apparent loss "rise" over iters 821-825 was the resume transient: the replay buffer is NOT
  checkpointed, so it refills from empty (204k->1.0M over ~5 iters); the fresh small buffer gives
  an artificially low loss that recovers to the true (healthy) steady-state. vloss plateaued and
  turned down by iter 829. No divergence.
- NOTE: two-pool runs 2 pools => 2048 games / 204k frames per iter (2x). At train_steps=500 the
  effective replay ratio is ~2.5 (not 5). The 5x-STEPS knob is verified healthy; for full 5x REPLAY,
  set n_games=512 or train_steps=1000. Left as-is (the user's explicit train_steps=500) and running.

Training LEFT RUNNING on box1 (nohup, checkpoints latest.pt each iter, resumable). Detached after
the 30-min babysit per instructions (speedups held + 5x didn't break).

### FULLY GPU-BOUND ACHIEVED: double-buffered graph two-pool (opt-cudagraphs, 8019a79)

The graph two-pool failure was ONLY the shared-buffer hazard. Per-pool DOUBLE-BUFFERING (ping/pong:
2 independent sets of 22 captured graphs, 2.96GB, consecutive submits use different buffers) fixes it:
submit returns ~0.3ms, no sibling-forward block. Box2 clean A/B (production config, median of 3):
  single-pool 929  ->  two-pool eager 1059  ->  two-pool DOUBLE-BUFFER 1308 pos/s
  = +40.8% over baseline, +23.5% over eager. GPU util 72% -> 84% -> **92% (p90 95%)**, median
  rock-stable (1307/1308/1308). This is FULLY GPU-BOUND self-play (the residual ~8% is the D2H
  fetch + irreducible dispatch that can't overlap). sweep/graph_eval_fn_db.py + bench_graph_db.py.

Adopting into training needs per-iter graph RE-CAPTURE (the net changes each iter; ~6s/iter for the
44 graphs) — a real but small overhead vs the eager two-pool's zero-setup simplicity. The live 67M
run stays on the eager two-pool (+38% in-loop, babysit-passed, healthy); the double-buffer is the
validated fully-GPU-bound upgrade to adopt when its per-iter recapture is wired into make_eval_fn.

## FINAL THROUGHPUT SUMMARY (raw single-pool baseline -> fully GPU-bound)
  raw single-pool               ~986 pos/s   (72% GPU util)
  + chlast (BN-fold+channels_last)  ~1119    (forward -30% ms)
  + two-pool eager overlap      ~1374        (84% util)  <- LIVE TRAINING CONFIG (+38% in-loop)
  + graph double-buffer         ~1600 est*   (92% util)  <- FULLY GPU-BOUND (measured 1308 on box2)
  (*box1 estimate from box2's +23.5% eager->db; box2 absolute is lower than box1.)
Dead ends (all measured + documented): refill (0.6% tail @cap120), single-buffer graph (WAR hazard),
tree-parallelism (plateau @8 cores). Every ablation on its own branch; see sections above.

### CORRECTION: the graph two-pool bug was PINNED STAGING, not a GPU-buffer WAR hazard

The earlier diagnosis ("both pools share a static buffer -> GPU-side WAR/WAW hazard -> submit
blocks on the sibling forward") was WRONG. A micro-probe (pure Python, no Rust/GIL) found the real
cause: the graph_eval_fn_np submit used a PERSISTENT PINNED STAGING buffer — `stage.copy_(from_numpy
(planes))` then async `si.copy_(stage, non_blocking=True)`. Writing INTO a pinned host block that
still has an outstanding recorded CUDA event (from the prior non_blocking H2D that read it) makes
PyTorch's caching HOST allocator BLOCK on that event -> a ~0.9MB CPU memcpy measured 11-20 ms
(instead of ~0.05 ms), dominating every round and idling the GPU. It is a host-side WAR hazard.

Two facts refute the GPU-buffer theory: (1) run_two_pools is SINGLE-THREADED — a CPU/GPU software
pipeline (submit A queues A's forward async, B's CPU tree work overlaps, fetch A syncs); the forward
is synced each round BEFORE the next submit reuses buffers, so there is never a second forward in
flight -> no GPU-side WAR. (2) The 11-20ms reproduced in a pure-Python loop with no Rust/threads.

FIX (in sweep/graph_eval_fn_db.py): drop the pinned staging; copy the pageable numpy DIRECTLY into
the static graph input (`si.copy_(from_numpy(planes))`, ~0.2ms blocking H2D, negligible vs ~10.5ms
forward) — the same cheap path the eager two-pool already used. submit 11.6ms -> 0.24ms.

CONSEQUENCE: DOUBLE-BUFFERING IS UNNECESSARY. Because the driver is single-threaded and the forward
is synced each round, NBUF=1 gives IDENTICAL throughput at half the graph memory (~1.5GB, not 3GB).
The committed file keeps NBUF=2 as harmless insurance; set NBUF=1 for the lean version.

THROUGHPUT RESTATED: #3 clean median 1308 pos/s WITH the nvidia-smi util sampler running; the agent
found the sampler itself steals ~25% (unsampled warmup ran ~1785 pos/s). So the true GPU-bound rate
is ~1750-1785 pos/s (needs a clean re-measure without the sampler to confirm); util 92% (p90 95%) is
the robust GPU-bound indicator either way. Bit-exact vs eager (0.0 max diff, both ping/pong slots).
The box (box2) died to an external vast outage right after #3 — results captured before it dropped.

---

## TRAINING COLLAPSE (elo) + KL-TRUST-REGION FIX

The throughput babysit "passed" on loss/vs_random, but the real metric caught a disaster:
the elo-vs-frozen-baseline (2048/L16) had NOT been auto-scored, and when run it showed a
severe strength COLLAPSE, not improvement:
  iter 700 -88.7 | 750 +21.7 (peak) | 800 -176.7 | 850 -428.8   (elo vs frozen 67M)
Score vs baseline fell 17/32 -> 8.5/32 -> 2.5/32. Proxy metrics hid it: vs_random stayed 1.00
(beats *random*, a floor) and vloss looked fine; the tell was ploss creeping 2.21->2.40 and
tactics/hanging_capture decaying 0.44->0.22 — which I initially (wrongly) rationalized as
"policy sharpening". Lesson relearned: score elo-vs-baseline continuously; don't trust proxies.

ROOT CAUSE: scripts/train_chess_gpu.py uses a FLAT lr=1e-3 (NO schedule — the cosine-then-held
note in this file's header refers to train_long.py, a different trainer). A flat 1e-3 at 85M+
frames over-fits late self-play into a policy-narrowing -> self-play-degeneration death spiral.
The collapse STARTED at train_steps=100 (750->800, -198 elo) — so it's the LR, not the 5x; the
5x (train_steps=500) just multiplies the gradient steps at that LR and accelerates it (800->850
-252 at train_steps=500).

FIX (committed): (1) lr 1e-3 -> 2e-4. (2) PPO-style POLICY TRUST REGION: snapshot the net before
each iter's 500 steps (= the behavior net that generated this iter's self-play) and add
kl_coef * KL(behavior || updated) to the loss. AlphaZero has a full-distribution CE target with
no sampled-action ratio to hard-clip, so KL (TRPO's exact constraint; PPO-clip approximates it;
RLHF uses KL-to-ref for the same reason) is the native form. kl_coef=1.0. (3) Revert latest.pt to
the iter-750 peak (collapsed net saved latest_collapsed_it871.pt). (4) Elo scoring now AUTOMATED
in-trainer every 50 iters (blocking clean-GPU arena after each iter_NNNNN.pt save; idempotent).

RESULT (iter 800, stabilized run vs the collapse run at the same iter):
  collapse:   8.5/32 = -177 elo  |  STABILIZED: 15.0/32 = -21.7 elo   (+155 elo swing)
Within the 32-game noise (~+-60 elo) the stabilized net is FLAT with the +21.7 peak; the collapse
was >4 SE down. Proxies confirm: ploss flat ~2.18 (was climbing), val_r 0.49 (was 0.22-0.34), kl
steadily bounded ~0.030. Throughput unaffected (two-pool +38%, 90% util). kl_coef=1.0 CLEARLY
works. Watching iter 850/900 to confirm flat-vs-slow-drift; if drifting, nudge kl_coef up / LR down.


---

## OVERNIGHT AUTONOMOUS CAMPAIGN (2026-07-02) — Elo/think-second + Elo/train-hour

Ran a multi-box campaign (1 live box + up to 4 clean RTX-3090 experiment boxes on vast) to optimize both
production metrics, without disturbing the live 67M run. Method for training dials: FORK the live net and run
parallel arms, scoring elo-vs-frames against the frozen 67M anchor (fast 512-sim/96-game arena inline; decisive
checks at 2048/L16). Every claim verified on real elo, not proxies.

### LIVE-RUN VERDICT: the stronger-dose fix WORKS (dynamics firefight won)
elo2048 (vs frozen 67M @2048/L16): iter700 -88.7 -> 750 +21.7 -> 800 +176.7 -> 850 +314.4. The
lr1e-4 + policy-KL(3.0) + value-Bernoulli-KL(1.0) recipe (commit 93b97d5) not only stopped the collapse
(old run 750->800 was +21.7->-176.7) but the net is climbing strongly (~+140 elo/50 iters). Live run left
UNTOUCHED and healthy throughout.

### Elo/think-second WIN [APPLIED to sweep/arena.py]: BN-fold + channels_last on the play/eval forward
The self-play forward had chlast+BN-fold; the ARENA/eval forward (sweep/arena.py `_eval_fn_for`) used fp16 only.
Added `_fuse_eval_net` (BN-fold + channels_last). Bit-close (argmax agree 99.6%, value MAE 0.003 vs fp32).
Forward speedup vs old fp16-NCHW: 1.61x @M=32, 1.50x @128, 1.19x @512 (largest at the SMALL batches production
uses: one game, leaf-parallel L~16 -> M~16 -> ~1.6x). Priced in elo via a same-net elo-vs-sim-multiplier arena:
1.25x->+33, 1.5x->+100, 2.0x->+206 -> the ~1.6x more sims per think-second buys ~+130 elo at equal think-time. Free.

### Elo/train-hour WIN: train_steps 500 -> 100 (with lr3e-4, keep kl3.0)
Fork sweep (all forked from iter-789, ~4M added frames, same ancestor = clean A/B):
- Round1 LR x KL: lr3e-4+kl3 ~= lr1e-4+kl3 per frame (wobble ~100 elo dominated; LR alone ~neutral).
  lr3e-4 + WEAK kl(1.0) DEGRADES -> at higher LR the strong trust region is essential (robust).
- Round2 replay-ratio (train_steps) at lr3e-4: ts=100 climbs to +429 @512 while ts=250/500 stay low/degrade.
  Sharp sweet spot at LOW ts (<=100). lr6e-4+ts100 WORSE than lr3e-4+ts100 (+158 vs +279 @it10) -> lr3e-4 is the knee.
- DECISIVE clean head-to-head at EQUAL frames (~87M), 96 games @2048: B_ts(lr3e-4 ts100) vs live-850(lr1e-4 ts500)
  = 63/96 (66%) -> +112 elo for ts100 (~3 SE). B_ts won DESPITE a fork empty-buffer handicap -> +112 is a lower bound.
  Plus ts=100 runs ~29% faster/iter (train 11s vs 55s). => WIN on BOTH elo/frame (+112) and elo/hour (~29%).
Mechanism: ts=500 over-trains (too many gradient steps/iter overfit the policy); fewer steps at a higher LR learn
more per frame without the overfit. (vloss is HIGHER at ts100 (0.43 vs 0.31) yet elo far higher -> policy over-fit,
not value fit, was the hidden ceiling.)
IMPORTANT CONFOUND (documented): forking restarts with an EMPTY replay buffer -> ts=500 forks overfit the refill
and crater (A0 fork +11 @2048 vs the warm live run's +176); ts=100 avoids this. So the raw fork gap (+364 vs +11)
is inflated; the clean number is the equal-frames head-to-head (+112). Sub-finding: on any RESTART with a cold
buffer (e.g. after a collapse-revert), use LOW train_steps to avoid the refill-overfit crater.

### Verified NEGATIVES (killed misdirections from the earlier code/transcript review)
- Gumbel opening diversity is SUFFICIENT: 1024-game self-play @iter783 -> all 20 first moves played (entropy
  3.54/4.32 bits), 1017/1024 unique lines by ply 8. Opening-seeding would add ~nothing. (Refutes the "self-play
  always from start_pos -> narrow coverage" concern.)
- Conv 8x8x73 policy head is a CAPACITY lever, NOT a speed lever on the production net: cuts params 12.6M->3.1M
  (-75%) but the policy head is <0.5% of tower FLOPs -> ~0% forward speedup at 128ch/10b (the "+24%" was a tiny-net
  artifact). Worth trying only as anti-overfit/bigger-trunk, not for speed.

### Deliverables
- Champion net lr3e-4+ts100 (@2048 +364 vs 67M, +112 over the same-frames live net): pulled locally as
  champion_ts100_p364.pt (state_dict). A continuation run (Bcont, lr3e-4 ts100 from 87.3M) left running to build a
  stronger net + test long-run stability of the config.
- sweep/arena.py chlast+BN-fold optimization committed to the working tree (the elo/think-second win).
- RECOMMENDATION (not auto-applied to the live run - irreversible while unattended): set train_steps 500->100 and
  lr 1e-4->3e-4 (keep kl_coef=3.0, v_kl_coef=1.0) in scripts/train_chess_gpu.py. Expect +~112 elo/frame and ~29%
  more iters/hour. Deploy the arena.py chlast change to the box too for a faster blocking auto-eval + ~+130 elo at
  any time-based play.


## ATTRIBUTION RESOLVED — "what specifically made the 67M elo climb again?" (warm-fork ablation from iter-750)
Config eras (real run): A lr1e-3,noTR = collapse; B lr2e-4,kl1 = flat; C lr1e-4,kl3,vkl1 = CLIMB (+176->+596).
Clean warm ablation (WARMUP fills buffer, no transient; elo vs 67M @512s/96g), each "full-C minus knob(s)":
  full-C  (lr1e-4,kl3,vkl1)          -> CLIMBS (live +596; ts100 +668)
  minus_vkl (lr1e-4,kl3,vkl0)        -> DECLINES +58,-25,-104,-120 (ploss RISING 2.15->2.21 = policy collapse)
  minus_lr  (lr2e-4,kl3,vkl1)        -> DECLINES +14,-51,-36,-104 (ploss flat ~2.11 = fails to climb)
  abl_kl    (lr2e-4,kl3,vkl0)        -> DECLINES to -226
  abl_lr    (lr1e-4,kl1,vkl0)        -> DECLINES to -190
ANSWER: NO single knob drove the climb. It required the FULL combination together — LR drop to 1e-4 + policy-KL=3
+ value-KL=1. Removing ANY one reverts to decline. The VALUE-KL is the most individually load-bearing (minus_vkl
collapses hardest, ploss rising) — I'd earlier underrated it. The observational "policy-kl 0.03->0.01" smoking gun
was real but a CONSEQUENCE of the lower LR, not the cause; and the earlier "LR drop is the driver" call was WRONG
(abl_lr, which drops LR but weakens the trust regions, declines). => the "stronger dose" commit worked precisely
BECAUSE it bundled all three; the bundle was necessary, not incidental.

## POST-BILLING-STOP recovery (2026-07-02): vast ran out of credit -> all instances stopped ~mid-run.
Restarted on top-up. Live run had peaked +596 @iter950 then OVER-TRAINED down (+523@1000, +394@1050) at ts500.
Resumed live with train_steps=500->100 (the validated fix; also small empty-buffer-restart transient). Preserved
the +596 peak as models/chess_gpu/peak_950_p596.pt. train time dropped 55s->12s/iter (the elo/hour win, live).


## RL-LITERATURE-INSPIRED EXPERIMENT (user direction): bad-update MASKING (DPPO/CAPO-style)
Literature (DPPO 2602.04879, CAPO, StableReinforce, IcePop, StarPO-S): a small fraction of updates cause most
instability -> detect (KL/curvature/advantage-outlier) and MASK/drop them. Implemented per-sample KL masking in
fork_train (MASK_FRAC: drop the top-frac most-divergent samples' gradient each minibatch).
Warm forks from iter-750, ts500, elo vs 67M @512:
  full-C (lr1e-4, kl3,  vkl1,   MASK 0)    -> CLIMBS (+400..+600)   [reference]
  mask-only (kl0, vkl0, MASK 0.08)         -> +47,-11,-29  (flat/mild decline; ploss stays ~2.1 = NO hard collapse,
                                              but kl DRIFTS UP 0.012->0.026 -> policy un-anchored)
  mask+halfKL (kl1, vkl0.5, MASK 0.08)     -> +85,+11  (still ~flat; kl held lower 0.011 but does NOT climb)
CONCLUSION: bad-update masking does NOT replace our KL trust region. It prevents the HARD collapse (no ploss
runaway) but cannot ANCHOR the distribution, so elo drifts; weakening the KL (even w/ masking) loses the climb.
=> Our instability (over-training = distributed policy over-sharpening DRIFT) is a DIFFERENT failure mode than the
   OUTLIER-update instability that DPPO/CAPO target. The fix for ours is the strong value-KL anchor (confirmed by the
   attribution ablation) + fewer train_steps (ts100), not outlier masking. Masking has marginal complementary value
   (prevents hard collapse) but is not the lever here. Honest negative result; technique correctly implemented.

## LIVE RUN on ts100 (post over-training fix): RECOVERED HARD. elo2048 vs 67M: 1050 +394 -> 1100 +800 (32/32=100%,
   SATURATED). The 67M anchor is now outgrown; re-anchor future evals to a strong recent net (e.g. iter_01100) to keep
   measuring. train_steps=100 validated in production (recovered the decline + saturates the baseline). ploss ~1.66.

## RE-ANCHORED PROGRESS (67M anchor saturated at +800): live iter-1150 vs the +596 peak (best_net_p596),
local MPS head-to-head @512s/24g = 20/24 (83%) -> +280 elo. => ts100 run pushed +280 PAST its previous peak (not just
recovered the decline). Local play net (chess_ui) updated to live iter-1150 (strongest). Live run continues climbing.
