# AlphaZero from scratch

A from-scratch AlphaZero implementation, built as a research project. The search,
self-play, and training loop are **game-agnostic**; chess is the main testbed, and the
same code trains other games (3D Connect 4 is in progress).

The chess net was trained entirely by self-play — no human games, no opening books, no
tablebases — and reached roughly **~1770 blitz on Lichess**, undefeated against the
human-calibrated Maia bot ladder.

**▶ Play it in your browser:** the trained net runs fully client-side (ONNX Runtime Web +
a JavaScript MCTS, no server-side compute) at **https://sheggle.com/chess-bot**.

---

## What's interesting here

- **Gumbel MCTS** (Danihelka et al., 2022) instead of vanilla PUCT. The stored policy
  target is *not* visit counts — it's the Gumbel completed-Q improved policy
  `pi = softmax(logits + sigma(completedQ))`, which is far more sample-efficient at the
  low simulation counts used during self-play.
- **Rust movegen (`fastchess/`)** via PyO3. Python chess dynamics were the throughput
  bottleneck (GPU idling at 0%); moving movegen + board→tensor encoding + leaf-parallel
  PUCT into Rust, with batched net inference on the GPU, made self-play GPU-bound.
- **Throughput engineering**: BN-folding + channels-last (NHWC tensor cores), fp16,
  CUDA-graph capture, and double-buffered two-pool overlap to keep the GPU saturated
  during single-game leaf-parallel search. See `CHESS_LOG.md` for the full running log.
- **Time-based, production-faithful evaluation**: strength is measured as Elo per unit of
  think-time, one game at a time with leaf/virtual-loss parallelism — the way the engine
  would actually be used — rather than at a fixed simulation count.
- A **128-channel / 10-block ResNet** over an 18-plane board encoding, with a 4672-action
  move head (from-square × 73 move-planes).

## Layout

| Path | What |
|------|------|
| `alphazero/` | Game-agnostic search (`mcts`, `az_mcts`, `gumbel`) + per-game env / encoding / net / training. Chess (`chess_*`) and 3D Connect 4 (`connect4_*`). |
| `fastchess/` | Rust chess engine (movegen, encoding, leaf-parallel PUCT, batched self-play) exposed to Python via PyO3. |
| `chess_ui/` | Play against the net: a local server, a Lichess bot, and a repetition-aware play wrapper. |
| `scripts/` | Training entry points (`train_chess_gpu.py`, `train_connect4.py`). |
| `sweep/`, `tests/`, `*.py` benches | Evaluation, hyperparameter sweeps, and throughput benchmarks. |
| `CHESS_LOG.md` | The full research log — every decision, dead end, and speedup, in order. |

## Quickstart

```bash
# Python env (uses uv)
uv sync

# Build the Rust engine
cd fastchess && uv run maturin develop --release && cd ..

# Train chess (GPU)
uv run python scripts/train_chess_gpu.py

# Play the net locally
uv run python chess_ui/play_server.py   # then open http://localhost:8088
```

## A note on style

Because the value target is the raw game outcome (win = +1 regardless of *how* it's won),
the net is material-light and activity-first: it will happily give up pawns to open a
position and pile on pressure. The flip side — near a clearly won position the value head
saturates and it can struggle to *convert* — is discussed (with a fix) in `CHESS_LOG.md`.

---

Built with [Claude Code](https://claude.com/claude-code). This is a research repository —
expect a running-log style rather than polished library code.
