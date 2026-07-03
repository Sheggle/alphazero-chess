"""GPU throughput benchmark for the Rust batched self-play engine.

Runs `fastchess.run_selfplay` with the 64ch/6b net, fp16 forward, at several
batch widths (n_games). Reports self-play positions/sec (training samples = moves
played), leaf inferences/sec, the fraction of wall time inside the net forward
(vs Rust tree ops + masking), and mean GPU utilisation.

SSH stdout for bg jobs is flaky -> always writes results to --out (default
bench_rust_results.txt); cat that file separately.

Usage: python bench_rust_selfplay.py [--out FILE] [--batches 256,512,1024,2048]
                                      [--sims 32] [--mc 16] [--max-ply 80]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
import pathlib

import numpy as np
import torch

_so = str(pathlib.Path(__file__).resolve().parent / "fastchess" / "pybuild")
if _so not in sys.path:
    sys.path.insert(0, _so)
import fastchess  # noqa: E402

from alphazero.chess_net import ChessNet  # noqa: E402


class GpuSampler(threading.Thread):
    def __init__(self, period=0.2):
        super().__init__(daemon=True)
        self.period = period
        self.stop_flag = False
        self.utils = []
        self.mems = []

    def run(self):
        while not self.stop_flag:
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"], text=True).strip().splitlines()[0]
                u, m = out.split(",")
                self.utils.append(float(u))
                self.mems.append(float(m))
            except Exception:
                pass
            time.sleep(self.period)


def build_eval_fn(net, device, fp16):
    state = {"fwd_time": 0.0, "n_eval": 0, "n_calls": 0}

    @torch.no_grad()
    def eval_fn(planes, legal_rows, legal_cols):  # planes (B,18,8,8) f32; legal_* (M,) int64
        t0 = time.perf_counter()
        x = torch.from_numpy(planes).to(device, non_blocking=True)
        if fp16:
            x = x.half()
        logits, values = net(x)
        logits = logits.float()
        # Gather only the legal logits on-GPU -> D2H ~B*35 floats, not B*4672.
        r = torch.from_numpy(legal_rows).to(device, non_blocking=True)
        c = torch.from_numpy(legal_cols).to(device, non_blocking=True)
        legal_logits = logits[r, c].contiguous().cpu().numpy()
        values = values.float().contiguous().cpu().numpy()
        if device == "cuda":
            torch.cuda.synchronize()
        state["fwd_time"] += time.perf_counter() - t0
        state["n_eval"] += planes.shape[0]
        state["n_calls"] += 1
        return np.ascontiguousarray(legal_logits, dtype=np.float32), np.ascontiguousarray(values, dtype=np.float32)

    return eval_fn, state


def run_one(net, device, fp16, n_games, sims, mc, max_ply, out_lines):
    eval_fn, state = build_eval_fn(net, device, fp16)
    # warmup the kernels at this batch width
    with torch.no_grad():
        wx = torch.zeros(n_games, 18, 8, 8, device=device)
        if fp16:
            wx = wx.half()
        net(wx)
        if device == "cuda":
            torch.cuda.synchronize()

    sampler = GpuSampler()
    sampler.start()
    t0 = time.perf_counter()
    samples, stats = fastchess.run_selfplay(
        eval_fn, n_games, sims, mc, 50.0, 1.0, 1.5, max_ply, 1.0, True, 12345)
    wall = time.perf_counter() - t0
    sampler.stop_flag = True
    sampler.join(timeout=1.0)

    n_pos = len(samples)             # training positions = moves played
    n_leaf = state["n_eval"]         # leaf inferences
    fwd = state["fwd_time"]
    rust = wall - fwd
    util = float(np.mean(sampler.utils)) if sampler.utils else float("nan")
    mem = float(np.max(sampler.mems)) if sampler.mems else float("nan")
    avg_plies = float(np.mean([s["plies"] for s in stats]))
    avg_batch = n_leaf / max(1, state["n_calls"])

    line = (f"B={n_games:5d} | pos/s={n_pos/wall:8.0f} | inf/s={n_leaf/wall:9.0f} | "
            f"games={len(stats)} pos={n_pos} leaves={n_leaf} | wall={wall:6.1f}s | "
            f"fwd={100*fwd/wall:4.1f}% rust={100*rust/wall:4.1f}% | "
            f"avg_batch={avg_batch:6.0f} calls={state['n_calls']} | "
            f"GPUutil={util:4.1f}% mem={mem:.0f}MiB | avg_plies={avg_plies:.0f}")
    print(line, flush=True)
    out_lines.append(line)
    return n_pos / wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench_rust_results.txt")
    ap.add_argument("--batches", default="256,512,1024,2048")
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--mc", type=int, default=16)
    ap.add_argument("--max-ply", type=int, default=80)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = (device == "cuda") and not args.fp32
    torch.manual_seed(0)
    net = ChessNet(channels=args.channels, blocks=args.blocks).to(device)
    net.eval()
    if fp16:
        net = net.half()

    out_lines = []
    header = (f"=== Rust batched self-play | {args.channels}ch/{args.blocks}b | "
              f"device={device} fp16={fp16} | sims={args.sims} mc={args.mc} "
              f"max_ply={args.max_ply} ===")
    print(header, flush=True)
    out_lines.append(header)
    try:
        nv = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.used,utilization.gpu",
             "--format=csv,noheader"], text=True).strip()
        out_lines.append("GPU at start: " + nv)
    except Exception:
        pass

    batches = [int(x) for x in args.batches.split(",")]
    for b in batches:
        try:
            run_one(net, device, fp16, b, args.sims, args.mc, args.max_ply, out_lines)
        except Exception as e:
            msg = f"B={b}: ERROR {type(e).__name__}: {e}"
            print(msg, flush=True)
            out_lines.append(msg)

    pathlib.Path(args.out).write_text("\n".join(out_lines) + "\n")
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
