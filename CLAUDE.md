# Research project — AlphaZero (chess now; tic-tac-toe earlier)

## Working directives (follow these)
- **Delegate specific research to sub-agents.** If a research/optimization
  question is well-scoped enough to be "just execution" once framed, hand it to a
  sub-agent — do NOT run it inline. Too many parallel investigations pollute the
  main thread. Keep the main thread for architecture, integration, decisions, and
  the log. Lead/org model: specialists build & measure & report; the lead verifies,
  decides, integrates, and is the SOLE writer of the log.
- **Verify speedups on real end-to-end merit** (games/s, elo-vs-frames), never a
  micro-benchmark alone. A faster micro-op that doesn't move the real metric is
  rejected (e.g. fastchess was 23-499x on ops but CPU break-even end-to-end).
- **Board -> tensor:** build the representation as a CPU-contiguous blob (numpy or
  Rust) and copy to GPU ONCE (`torch.from_numpy(...).to(cuda)`). NEVER set tensor
  entries per-element, especially on GPU (each is a kernel launch — pathologically
  slow). This path is a prime throughput suspect; keep it out of the hot loop.
- **Eval must mirror ideal production, and production is TIME-BASED.** Production =
  MAXIMUM ELO PER THINK-TIME, ONE GAME AT A TIME, with leaf/node parallelism (virtual
  loss) filling the GPU *within that single game* — NOT game-batching, and NEVER
  batch-1 / L=1 (that path was deliberately removed). So eval uses a wall-clock
  think-time budget per move, never a fixed sim count. A net does as many sims as it
  can in the think-time, so fast nets legitimately get more search; a net that can't
  convert think-time into strength (e.g. value head OOD at the depth the time now
  affords) is GENUINELY production-weak — that's a correct penalty, not a confound to
  engineer away. Feasibility trick: calibrate sims = (single-game leaf-parallel
  nodes/s) x think_time per net, then run the tournament game-batched at that sim
  count (sim count = production; batching only for eval throughput). Always use a
  fixed established opening suite (UHO/TCEC-style), every opening played BOTH colors,
  deterministic best-play after the book (argmax visits, no sampling at eval).

## Where things are
- `CHESS_LOG.md` — running research log + every lead decision (read for state).
- `alphazero/`: game-agnostic search (mcts / az_mcts / gumbel); chess_env /
  chess_encode / chess_net / chess_train; chess_batched (bit-exact batched
  self-play = GPU lever); chess_env_fast + `fastchess/` (Rust movegen, rejected on
  CPU end-to-end). Best chess net: `models/chess_keeper/iter_00050.pt`.

## Algorithm gotchas (don't be shocked)
- **The stored policy target is NOT visit counts.** Each self-play sample's policy
  target (`mcts_probs` in training / `pi_val` in Rust) is the **Gumbel completed-Q
  improved policy**: `pi = softmax(logits + sigma(completedQ))` over legal moves.
  `completedQ[a] = q(a)` for visited actions, else `v_mix` (the Gumbel value
  completion `(root_value + n_total*wq)/(1+n_total)`, `wq` = prior-weighted mean Q of
  visited actions). `sigma(q) = (c_visit + max_visit)*c_scale*q` (c_visit=50,
  c_scale=0.3). Visit counts ONLY drive Sequential-Halving action selection
  (`gpref + sigma(Q)`), never the training target — this is the whole point of Gumbel
  (visit counts are a bad target at low sim counts). Code: `fastchess/src/selfplay.rs`
  `finalize()` (~L581-642) + `sigma()` (L331); the `_completed_policy` it mirrors is in
  `alphazero/gumbel.py`. Training CE pushes the net policy toward this pi.

## GPU box (vast.ai)
- `ssh -p <PORT> root@<VAST_BOX>` — RTX 3090, 72 cores, 128G. Bare base image:
  Python env `/venv/main` (`source /venv/main/bin/activate`; torch+cu13 installed
  there). Code synced to `/root/research`. SSH stdout capture is flaky for bg jobs
  — write results to a file on the box and cat it. Read `/etc/vast-agents-guide.md`
  before exposing services; user reaches forwarded ports via their `ssh -L` tunnel.

## Current focus (step 1: throughput)
- Maximize self-play **games/s** with the SAME algorithm (preserve elo-vs-frames).
  Bottleneck = Python chess dynamics + per-node board ops (GPU idles at 0% in
  single-process batched self-play; Python baseline ~1.6 games/s @32 procs).
  Direction: move chess dynamics + encoding OFF Python (Rust), batch net inference
  on GPU, parallelize search across the 72 cores.
