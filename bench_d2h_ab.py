"""Controlled A/B microbench of the per-round D2H cost: full (B,4672) logits
transfer vs the new legal-only on-GPU gather. The forward is IDENTICAL in both
paths, so the delta = exactly the D2H+gather cost the new engine removes -- and
the delta is robust to concurrent GPU load (both paths eat the same forward+sync
under the same contention). Reports per-round ms, payload bytes, and the
projected per-iteration (~2640 rounds) wall savings.
"""
import sys, time, pathlib
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "fastchess" / "pybuild"))
from alphazero.chess_net import ChessNet  # noqa: E402

AVG_LEGAL = 35   # ~mean legal moves/position in chess (matches bench leaves/pos)
N = 300


def main():
    dev = "cuda"
    torch.manual_seed(0)
    net = ChessNet(channels=64, blocks=6).to(dev).eval().half()
    out = []
    for B in (1024, 2048):
        x = torch.zeros(B, 18, 8, 8, device=dev).half()
        rows = np.repeat(np.arange(B), AVG_LEGAL).astype(np.int64)
        cols = np.random.randint(0, 4672, size=B * AVG_LEGAL).astype(np.int64)
        r = torch.from_numpy(rows).to(dev)
        c = torch.from_numpy(cols).to(dev)

        with torch.no_grad():
            for _ in range(30):
                logits, values = net(x)
                _ = logits.float().contiguous().cpu().numpy()
            torch.cuda.synchronize()

            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(N):
                logits, values = net(x)
                _ = logits.float().contiguous().cpu().numpy()
                _ = values.float().contiguous().cpu().numpy()
                torch.cuda.synchronize()
            t_full = (time.perf_counter() - t0) / N

            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(N):
                logits, values = net(x)
                lg = logits.float()
                _ = lg[r, c].contiguous().cpu().numpy()
                _ = values.float().contiguous().cpu().numpy()
                torch.cuda.synchronize()
            t_legal = (time.perf_counter() - t0) / N

        full_mb = B * 4672 * 4 / 1e6
        legal_mb = B * AVG_LEGAL * 4 / 1e6
        line = (f"B={B:5d} | full-logits D2H={full_mb:5.1f}MB t={t_full*1e3:6.2f}ms/round | "
                f"legal-gather D2H={legal_mb:6.3f}MB t={t_legal*1e3:6.2f}ms/round | "
                f"saved={ (t_full-t_legal)*1e3:5.2f}ms/round ({100*(t_full-t_legal)/t_full:4.1f}%) | "
                f"payload {full_mb/legal_mb:5.0f}x smaller | ~per-iter(2640) saved={(t_full-t_legal)*2640:5.1f}s")
        print(line, flush=True)
        out.append(line)
    pathlib.Path("bench_d2h_ab.txt").write_text("\n".join(out) + "\n")
    print("wrote bench_d2h_ab.txt", flush=True)


if __name__ == "__main__":
    main()
