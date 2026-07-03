# Ping-Pong Eval Report

Branch: `opt-pingpong`

## Design

`fastchess.run_selfplay(eval_fn, n_games, ...)` now runs two independent self-play pools by default. `n_games` is interpreted as games per pool, so production `n_games=1024` runs pool A with 1024 games and pool B with 1024 games. This keeps each network batch full-width while providing independent CPU tree work during the other pool's forward. Returned samples/stats are concatenated as pool A then pool B.

The steady-state schedule is:

```text
submit(A_next_batch)
apply B_fetched_result + descend/build B_next_batch while A forward runs
fetch(A_result)

submit(B_next_batch)
apply A_fetched_result + descend/build A_next_batch while B forward runs
fetch(B_result)
```

Startup has one unavoidable unhidden fetch for the first submitted root batch. Drain can also lose overlap after one pool finishes. In the long production run those edges should be small.

The implementation keeps exactly one submitted eval handle live. It never submits B until A has been fetched, and vice versa, so only one forward's activations/workspace are live on the GPU. The second pool doubles CPU-side game/search state and can hold one prepared host batch plus one CPU result, but it does not double GPU forward memory.

For local validation only, `FASTCHESS_SINGLE_POOL_BASELINE=1` forces the old single-pool scheduler. This is not needed by the benchmark/trainer path.

## Eval Contract

`fastchess.run_selfplay` accepts either the existing synchronous callable:

```python
eval_fn(planes, legal_rows, legal_cols) -> (legal_logits, values)
```

or a two-phase evaluator object:

```python
handle = eval_fn.submit(planes, legal_rows, legal_cols)
legal_logits, values = eval_fn.fetch(handle)
```

`submit(planes, legal_rows, legal_cols) -> handle`

- `planes`: NumPy `float32`, shape `(B, 18, 8, 8)`, contiguous.
- `legal_rows`: NumPy `int64`, shape `(M,)`, contiguous.
- `legal_cols`: NumPy `int64`, shape `(M,)`, contiguous.
- Enqueues H2D, forward, and legal-logit gather on the device stream when the backend supports it.
- Must keep all tensors/storage needed by queued work alive through `handle`.
- Should not do result D2H synchronization.

`fetch(handle) -> (legal_logits, values)`

- Synchronizes the submitted work for that handle if needed.
- Returns `legal_logits`: NumPy `float32`, shape `(M,)`, contiguous, in exactly the submitted legal row/col order.
- Returns `values`: NumPy `float32`, shape `(B,)`, contiguous.
- Handles are consumed one at a time. Current Rust never has more than one live handle.

`scripts/train_chess_gpu.py::make_eval_fn` and `sweep/bench_selfplay.py` expose this contract. Their `submit` returns a handle containing the live input/index/logit/gather/value tensors; `fetch` performs `legal_logits.float().cpu().numpy()` and `values.float().cpu().numpy()`. `__call__` remains as `fetch(submit(...))` for compatibility.

## Local Verification

Build and install:

```text
cd fastchess && maturin build --release
mkdir -p pybuild
find pybuild -maxdepth 1 \( -name fastchess -o -name 'fastchess-*.dist-info' \) -exec rm -rf {} +
(cd pybuild && unzip -oq ../target/wheels/fastchess-0.1.0-cp312-abi3-macosx_11_0_arm64.whl)
```

Rust/Python checks:

```text
cargo check --release
uv run python -m py_compile scripts/train_chess_gpu.py sweep/bench_selfplay.py
```

CPU correctness check with `ChessNet(64, 4)`, `sims=8`, `max_ply=60`, `add_noise=True`, `seed=20260630`:

```text
two-pool sync callable: samples=1830 stats=32 time=11.050s
two-pool submit/fetch: samples=1830 stats=32 time=10.951s
samples_bitmatch=True
stats_equal=True
OK two-pool sync callable and submit/fetch are bit-identical on local CPU
```

Single-pool 32-game baseline versus two pools of 16 games:

```text
single-pool baseline 32: samples=1830 stats=32 time=7.895s
two-pool 2x16: samples=1830 stats=32 time=10.894s
stats_equal=True
planes_indices_z_bitmatch=True
max_policy_value_abs_diff=5.960464478e-08
```

The games, move stats, planes, policy indices, and value targets match. The only observed difference versus the old single-pool baseline is tiny policy-probability float drift from different network batch widths, which is expected. No CUDA timing was run.

## Overlap Estimate

At the measured production shape, the target is roughly:

```text
forward/fetch path: ~9.8 ms
CPU tree/resume/build path: ~5.7 ms
old serialized step: ~15.5 ms
ideal alternated step: max(9.8, 5.7) ~= 9.8 ms
```

This design can hide most of the CPU tree path because pool B's tree work uses B's already-fetched previous result and is independent of pool A's in-flight forward. If submit returns after enqueueing device work, the Rayon tree section should run while the GPU is busy.

The realistic critical path is closer to:

```text
submit dispatch + max(device forward/gather/D2H-ready wait, other-pool tree) + fetch D2H/numpy dispatch
```

The tree should hide well when it remains below the forward time, so recovering most of the measured ~5.7 ms idle is plausible. I would expect a clean-GPU production run to land much closer to the ~9.8 ms forward ceiling than to the old ~15.5 ms serialized round, but not exactly at the ceiling because:

- Python still holds the GIL during `submit` and `fetch` entry/exit.
- `fetch` contains the `.cpu().numpy()` D2H synchronization and NumPy materialization; that is on the critical path before the next pool can be submitted.
- Startup and drain are not overlapped.
- If a pool shrinks late in games, forward efficiency can fall and the other pool may not perfectly cover it.
- CPU contention doubles the number of live game trees, although only one pool's tree step is active during a forward.

The branch mitigates the previous failed double-buffer issues by keeping full-width batches, doing only one `submit` and one `fetch` per pool-step, and never allowing two GPU forwards/activation sets in flight.

