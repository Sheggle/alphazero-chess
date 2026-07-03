"""Throughput sweep for the RUST leaf-parallel PUCT arena (fastchess.arena_bench).

Per L: one move-step from the start position under a wall budget; report
nodes(sims)/s and the per-round split GPU-forward-ms (Python eval) vs Rust-tree-ms
(the entire MCTS: descents + virtual loss + encode + backups, in Rust). Compare
to the old Python arena where the tree cost was ~1ms/leaf (python-chess).

Inference is compiled+fp16, padded to a fixed batch=L (one CUDA-graph capture per
(net,L)). Writes to --out. SMAC sweep shares the GPU -> contended/conservative.
"""
from __future__ import annotations
import sys, time, argparse, copy
from pathlib import Path
import numpy as np, torch, chess

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "fastchess" / "pybuild")):
    if p not in sys.path:
        sys.path.insert(0, p)
import fastchess
from sweep.batched_arena import load_evaluator

DEV = "cuda"
SMALL = str(ROOT / "sweep" / "runs" / "cfg_001" / "final.pt")
BIG = str(ROOT / "models" / "chess_gpu" / "iter_00600.pt")
LS = [1, 2, 4, 8, 16, 32, 64, 128, 256]


class RustEval:
    """Padded, static-shape, fp16, (optionally compiled) eval for the Rust arena.
    __call__(planes (M,18,8,8) f32) -> (logits (M,4672) f32, values (M,) f32)."""
    def __init__(self, raw_net, device, L, mode=None, fp16=True):
        net = copy.deepcopy(raw_net).to(device)
        if fp16:
            net = net.half()
        net.eval()
        self.fp16, self.device, self.L = fp16, device, L
        self.net = torch.compile(net, mode=mode) if mode else net
        self.buf = torch.zeros(L, 18, 8, 8, device=device,
                               dtype=torch.float16 if fp16 else torch.float32)

    @torch.no_grad()
    def __call__(self, planes):
        n = planes.shape[0]
        t = torch.from_numpy(planes).to(self.device)
        if self.fp16:
            t = t.half()
        self.buf[:n] = t
        if n < self.L:
            self.buf[n:].zero_()
        logits, values = self.net(self.buf)
        return (np.ascontiguousarray(logits[:n].float().cpu().numpy(), dtype=np.float32),
                np.ascontiguousarray(values[:n].float().cpu().numpy(), dtype=np.float32))

    def warmup(self, iters=6):
        x = np.zeros((self.L, 18, 8, 8), dtype=np.float32)
        for _ in range(iters):
            self(x)
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="reduce-overhead")
    ap.add_argument("--ms", type=int, default=250)
    ap.add_argument("--ls", default="")
    args = ap.parse_args()
    mode = None if args.mode == "eager" else args.mode
    ls = [int(x) for x in args.ls.split(",")] if args.ls else LS
    lines = [f"### RUST ARENA THROUGHPUT (mode={args.mode}) — nodes/s + GPU-fwd vs Rust-tree split",
             "    (SMAC shares GPU; contended/conservative)"]
    for name, path in [("48ch small", SMALL), ("128ch big", BIG)]:
        ev = load_evaluator(path, DEV)
        lines.append(f"\n=== {name} | {args.ms}ms/move ===")
        lines.append(f"  {'L':>4} {'nodes/s':>9} {'gpu_ms/rnd':>11} {'rust_ms/rnd':>12} "
                     f"{'gpu%':>5} {'rounds':>7}")
        for L in ls:
            ce = RustEval(ev.net, DEV, L, mode=mode, fp16=True)
            try:
                ce.warmup(6)
            except Exception as e:
                lines.append(f"  L={L}: compile FAILED {type(e).__name__}: {e}")
                continue
            sims, rounds, tree_s, eval_s, wall = fastchess.arena_bench(
                ce, float(args.ms), 0, L, 1.5, 0)
            if rounds == 0:
                continue
            nps = sims / wall
            gpu_ms = eval_s / rounds * 1000
            rust_ms = tree_s / rounds * 1000
            gpu_pct = 100 * eval_s / (eval_s + tree_s)
            lines.append(f"  {L:>4} {nps:>9.0f} {gpu_ms:>11.3f} {rust_ms:>12.4f} "
                         f"{gpu_pct:>5.0f} {rounds:>7}")
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
