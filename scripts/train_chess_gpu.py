"""AlphaZero chess training on GPU, self-play via the Rust batched-MCTS engine.

Self-play: `fastchess.run_selfplay(eval_fn, ...)` runs thousands of concurrent
games and batches all leaf evals into one GPU forward per round (the `eval_fn`).
Training: standard policy-CE + value-MSE + entropy on the GPU. The GPU has spare
compute at the small-net throughput, so we use a bigger net here.

Run on the box:
  cd /root/research && source /venv/main/bin/activate
  PYTHONPATH=.:fastchess/pybuild python scripts/train_chess_gpu.py [smoke]
"""
import copy
import json
import os
import random
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fastchess" / "pybuild"))
import fastchess  # noqa: E402

from alphazero.chess_net import ChessNet, ChessEvaluator  # noqa: E402
from alphazero.chess_env import ChessGame  # noqa: E402
from alphazero.chess_encode import encode_board  # noqa: E402
from alphazero.chess_tactics import tactics_rates  # noqa: E402
import chess  # noqa: E402

ACTION_SIZE = 4672


def make_eval_fn(net, device, fp16):
    """Two-phase (submit/fetch) evaluator for the Rust TWO-POOL self-play engine.

    run_selfplay detects .submit/.fetch and runs pool-A-forward-on-GPU while
    pool-B does MCTS tree work on CPU (then swaps). The forward (net(x)) is queued
    in submit WITHOUT the D2H sync; the .cpu() is deferred to fetch, so pool B's
    tree work overlaps pool A's in-flight forward. Measured +22.8% over single-pool
    at 83.7% GPU util, sample-bit-exact. .__call__ is the synchronous fallback
    (single-pool builds ignore submit/fetch and call this).
    """
    class TwoPoolEval:
        @torch.no_grad()
        def submit(self, planes, legal_rows, legal_cols):  # planes (B,18,8,8) f32
            x = torch.from_numpy(planes).to(device, non_blocking=True)
            x = x.contiguous(memory_format=torch.channels_last)  # NHWC tensor-core, no transpose
            if fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    p, v = net(x)
            else:
                p, v = net(x)
            return {"p": p, "v": v, "rows": legal_rows, "cols": legal_cols}  # NO .cpu() yet

        @torch.no_grad()
        def fetch(self, h):
            # Gather only the legal logits on-GPU (Rust's per-game order), so only
            # ~B*35 floats cross back instead of the full (B,4672) tensor.
            r = torch.from_numpy(h["rows"]).to(device, non_blocking=True)
            c = torch.from_numpy(h["cols"]).to(device, non_blocking=True)
            legal_logits = np.ascontiguousarray(h["p"][r, c].float().cpu().numpy(), dtype=np.float32)
            values = np.ascontiguousarray(h["v"].float().cpu().numpy(), dtype=np.float32)
            return legal_logits, values

        def __call__(self, planes, legal_rows, legal_cols):
            return self.fetch(self.submit(planes, legal_rows, legal_cols))

    return TwoPoolEval()


def fused_inference_net(net):
    """Build an inference-only copy of the (BatchNorm, channels_last) training net
    for self-play: fold each BN into its preceding bias-free conv (mathematically
    exact in eval mode) and use channels_last memory. The self-play forward then
    runs with ~13% fewer kernels (no BN) and no NCHW<->NHWC transposes (~15%) — the
    measured +15.5% throughput win. Rebuilt each iter from the current weights; the
    *training* net keeps its BN (folding is eval-only, can't train through it)."""
    import copy
    from torch.nn.utils import fuse_conv_bn_eval
    m = copy.deepcopy(net).eval()
    m.stem[0] = fuse_conv_bn_eval(m.stem[0], m.stem[1]); m.stem[1] = torch.nn.Identity()
    for blk in m.tower:
        blk.c1 = fuse_conv_bn_eval(blk.c1, blk.b1); blk.b1 = torch.nn.Identity()
        blk.c2 = fuse_conv_bn_eval(blk.c2, blk.b2); blk.b2 = torch.nn.Identity()
    m.p_conv = fuse_conv_bn_eval(m.p_conv, m.p_bn); m.p_bn = torch.nn.Identity()
    m.v_conv = fuse_conv_bn_eval(m.v_conv, m.v_bn); m.v_bn = torch.nn.Identity()
    return m.to(memory_format=torch.channels_last)


def material_white(board):
    v = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    d = 0
    for _, p in board.piece_map().items():
        d += v.get(p.piece_type, 0) * (1 if p.color == chess.WHITE else -1)
    return d


def value_material_r(net, device, n=120, seed=0):
    rng = random.Random(seed)
    vs, ms = [], []
    net.eval()
    with torch.no_grad():
        for _ in range(n):
            b = chess.Board()
            for _ in range(rng.randint(6, 40)):
                if b.is_game_over():
                    break
                b.push(rng.choice(list(b.legal_moves)))
            if b.is_game_over():
                continue
            x = torch.from_numpy(encode_board(ChessGame(b))[None]).to(device)
            _, val = net(x)
            vs.append(float(val.item()))
            ms.append(material_white(b) * (1 if b.turn == chess.WHITE else -1))
    return float(np.corrcoef(vs, ms)[0, 1]) if len(vs) > 4 else 0.0


def vs_random(net, device, n_games=20, sims=64, mc=16, max_ply=160, seed=0):
    from alphazero.gumbel import GumbelMCTS
    from alphazero.chess_env import encode_move
    ev = ChessEvaluator(net, device=device)
    rng = np.random.default_rng(seed)
    rrng = random.Random(seed + 1)
    w = d = l = 0
    for i in range(n_games):
        net_white = (i % 2 == 0)
        g = ChessGame()
        while not g.is_terminal() and g.ply < max_ply:
            if (g.to_play == 1) == net_white:
                a, _ = GumbelMCTS(ev, n_sims=sims, max_considered=mc, c_scale=0.3,
                                  rng=rng).run(g, add_noise=False)
            else:
                legal = list(g.board.legal_moves)
                a = encode_move(g.board, rrng.choice(legal))
            g = g.apply(int(a))
        if g.is_terminal():
            res = g.result()
        else:
            md = material_white(g.board)
            res = 1 if md >= 1 else (-1 if md <= -1 else 0)
        nz = res if net_white else -res
        w += nz > 0; d += nz == 0; l += nz < 0
    return (w + 0.5 * d) / n_games, w, d, l


def train(smoke=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = dict(
        channels=128, blocks=10,
        n_games=1024, sims=64, eval_sims=32, mc=16, c_visit=50.0, c_scale=0.3, c_puct=1.5,
        max_ply=120, mat_thresh=3.0,  # "who wins": capped game decided only if a side is up a piece+
        buffer_size=1_000_000, batch_size=1024, train_steps=500,  # replay ratio ~5: mine the expensive self-play frames harder (self-play is the bottleneck, training is cheap)
        lr=1e-4, weight_decay=1e-4, entropy_coef=0.05,  # flat 1e-3 over-fit late self-play -> elo collapse; 2e-4 slowed but didn't stop it (elo still slid -100/100 iters) -> 1e-4
        kl_coef=3.0,   # policy trust region KL(behavior||updated); 1.0 cut the collapse 73% but the net still slid -> 3.0
        v_kl_coef=1.0,  # value trust region: Bernoulli KL on win-prob p=(v+1)/2, caps value-head over-fit (the 2048-sim elo leans on the value head)
        eval_every=10, ckpt_dir=str(ROOT / "models/chess_gpu"), iterations=100000,
    )
    if smoke:
        cfg.update(channels=64, blocks=6, n_games=256, train_steps=20,
                   eval_every=2, iterations=3, ckpt_dir=str(ROOT / "models/chess_gpu_smoke"))

    Path(cfg["ckpt_dir"]).mkdir(parents=True, exist_ok=True)
    metrics_path = Path(cfg["ckpt_dir"]) / "metrics.jsonl"
    torch.manual_seed(0); random.seed(0)
    torch.backends.cudnn.benchmark = True

    net = ChessNet(channels=cfg["channels"], blocks=cfg["blocks"]).to(device)
    net = net.to(memory_format=torch.channels_last)  # train + infer in NHWC (layout only; math unchanged)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    buffer = []  # plain list -> O(1) random-access sampling (deque sample is O(n))

    # Resume from latest.pt if present (reuse the net + optimizer; don't restart training).
    start_iter = 0
    start_frames = 0
    ckpt_path = Path(cfg["ckpt_dir"]) / "latest.pt"
    if ckpt_path.exists():
        st = torch.load(ckpt_path, map_location=device)
        net.load_state_dict(st["state_dict"])
        if "opt" in st:
            opt.load_state_dict(st["opt"])
        start_iter = int(st.get("iter", 0))
        start_frames = int(st.get("frames", 0))
        print(f"RESUMED from {ckpt_path} at iter {start_iter} ({start_frames/1e6:.2f}M frames)", flush=True)

    nparams = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"GPU trainer: {nparams:.1f}M params, {cfg['channels']}ch/{cfg['blocks']}b, "
          f"n_games={cfg['n_games']} sims={cfg['sims']} fp32-train device={device}", flush=True)

    games_total = start_iter * cfg["n_games"]
    frames_total = start_frames
    for it in range(start_iter + 1, cfg["iterations"] + 1):
        t0 = time.time()
        net.eval()
        # Fold BN + channels_last into a throwaway inference net from the current
        # weights (cheap: ~12.6M params copied once per ~100s iter). Self-play only.
        eval_fn = make_eval_fn(fused_inference_net(net), device, fp16=(device == "cuda"))
        samples, stats = fastchess.run_selfplay(
            eval_fn, cfg["n_games"], cfg["sims"], cfg["mc"], cfg["c_visit"],
            cfg["c_scale"], cfg["c_puct"], cfg["max_ply"], cfg["mat_thresh"], True, it)
        sp_t = time.time() - t0
        buffer.extend(samples)
        if len(buffer) > cfg["buffer_size"]:
            del buffer[:len(buffer) - cfg["buffer_size"]]
        games_total += len(stats)
        frames_total += len(samples)
        pos_s = len(samples) / sp_t

        # --- train ---
        net.train()
        ploss = vloss = kloss = vkloss = 0.0; nb = 0
        if len(buffer) >= cfg["batch_size"]:
            tt = time.time()
            # Behavior policy for the trust region: the net BEFORE this iter's updates
            # (= the net that generated this iter's fresh self-play). Frozen, no grad.
            ref_net = copy.deepcopy(net).eval()
            for p in ref_net.parameters():
                p.requires_grad_(False)
            for _ in range(cfg["train_steps"]):
                idxs = np.random.randint(0, len(buffer), size=cfg["batch_size"])
                batch = [buffer[i] for i in idxs]
                planes = torch.from_numpy(
                    np.stack([b[0] for b in batch]).astype(np.float32)).to(device)
                # vectorized sparse->dense policy target: one indexed scatter (no Python loop)
                rows = np.concatenate([np.full(len(b[1]), k, dtype=np.int64)
                                       for k, b in enumerate(batch)])
                cols = np.concatenate([b[1].astype(np.int64) for b in batch])
                vals = np.concatenate([b[2] for b in batch]).astype(np.float32)
                tpi = torch.zeros(len(batch), ACTION_SIZE, device=device)
                tpi[torch.from_numpy(rows).to(device),
                    torch.from_numpy(cols).to(device)] = torch.from_numpy(vals).to(device)
                tz = torch.from_numpy(
                    np.array([b[3] for b in batch], dtype=np.float32)).to(device)
                logits, value = net(planes)
                logp = F.log_softmax(logits, dim=1)
                pl = -(tpi * logp).sum(dim=1).mean()
                vl = F.mse_loss(value, tz)
                ent = -(logp.exp() * logp).sum(dim=1).mean()
                # PPO-style trust region: KL(behavior || updated) keeps the policy from
                # drifting too far from the net that generated the data (over the 500
                # steps) -> caps per-iter policy movement, the elo-collapse mechanism.
                with torch.no_grad():
                    ref_logits, ref_value = ref_net(planes)
                    ref_logp = F.log_softmax(ref_logits, dim=1)
                kl = (ref_logp.exp() * (ref_logp - logp)).sum(dim=1).mean()
                # Value trust region: treat the scalar value v in [-1,1] as a Bernoulli
                # win-prob p=(v+1)/2 over {loss=0, win=1}, and penalize KL(behavior || updated).
                # Limits how fast the value head over-fits/sharpens per iter (the 2048-sim
                # elo leans hard on the value; policy-only KL can't catch value drift).
                e = 1e-6
                p_new = ((value + 1) * 0.5).clamp(e, 1 - e)
                p_ref = ((ref_value + 1) * 0.5).clamp(e, 1 - e)
                vkl = (p_ref * (p_ref.log() - p_new.log())
                       + (1 - p_ref) * ((1 - p_ref).log() - (1 - p_new).log())).mean()
                loss = (pl + vl - cfg["entropy_coef"] * ent
                        + cfg["kl_coef"] * kl + cfg["v_kl_coef"] * vkl)
                opt.zero_grad(); loss.backward(); opt.step()
                ploss += pl.item(); vloss += vl.item(); kloss += kl.item(); vkloss += vkl.item(); nb += 1
            train_t = time.time() - tt
        else:
            train_t = 0.0
        ploss = ploss / nb if nb else float("nan")
        vloss = vloss / nb if nb else float("nan")
        kloss = kloss / nb if nb else float("nan")
        vkloss = vkloss / nb if nb else float("nan")
        dec = sum(1 for s in stats if s.get("z_white", 0) != 0)
        rec = {"iter": it, "games": games_total, "frames": frames_total, "buffer": len(buffer),
               "pos_s": round(pos_s), "sp_s": round(sp_t, 1), "train_s": round(train_t, 1),
               "ploss": round(ploss, 4), "vloss": round(vloss, 4), "kl": round(kloss, 4),
               "vkl": round(vkloss, 4), "decisive": dec, "n_games": len(stats)}

        if it % cfg["eval_every"] == 0:
            net.eval()
            ev = ChessEvaluator(net, device=device)
            rec["tactics"] = tactics_rates(ev, sims=cfg["eval_sims"], max_considered=cfg["mc"])
            rec["value_r"] = round(value_material_r(net, device, seed=it), 3)
            sc, w, d, l = vs_random(net, device, seed=it)
            rec["vs_random"] = {"score": round(sc, 3), "w": w, "d": d, "l": l}
            torch.save({"state_dict": net.state_dict(), "opt": opt.state_dict(),
                        "channels": cfg["channels"], "blocks": cfg["blocks"], "iter": it,
                        "frames": frames_total},
                       Path(cfg["ckpt_dir"]) / "latest.pt")
            if it % (cfg["eval_every"] * 5) == 0:
                torch.save({"state_dict": net.state_dict(), "channels": cfg["channels"],
                            "blocks": cfg["blocks"], "iter": it},
                           Path(cfg["ckpt_dir"]) / f"iter_{it:05d}.pt")
                # Elo-vs-frozen-baseline scoring on the just-saved checkpoint (every 50
                # iters). The cheap inline evals (vs_random/tactics) hid a real strength
                # collapse; this is the ground-truth signal. Blocking + clean-GPU (self-
                # play paused during the ~10min arena) so the elo is trustworthy; the
                # eval script is idempotent (scores only the new iter_NNNNN.pt) and
                # non-fatal (a failure never kills training).
                if device == "cuda" and not smoke:
                    try:
                        print(f"[elo] scoring iter {it} vs frozen 67M baseline (2048/L16)...", flush=True)
                        subprocess.run(
                            [sys.executable, str(ROOT / "sweep" / "eval_vs_baseline.py")],
                            cwd=str(ROOT),
                            env={**os.environ,
                                 "PYTHONPATH": f"{ROOT}{os.pathsep}{ROOT / 'fastchess' / 'pybuild'}"},
                            timeout=2400, check=False)
                    except Exception as e:  # never let eval crash training
                        print(f"[elo] scoring failed (non-fatal): {e}", flush=True)

        print(
            f"it {it:4d} | {frames_total/1e6:.1f}M frames | games {games_total:7d} | {pos_s:.0f} pos/s "
            f"(sp {sp_t:.1f}s tr {train_t:.1f}s) | buf {len(buffer)} | "
            f"ploss {ploss:.3f} vloss {vloss:.3f} kl {kloss:.4f} vkl {vkloss:.4f} | dec {dec}/{len(stats)}"
            + (f" | tactics {rec['tactics']['overall']:.2f} val_r {rec['value_r']} "
               f"vs_rand {rec['vs_random']['score']:.2f}" if "tactics" in rec else ""),
            flush=True)
        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    train(smoke=(len(sys.argv) > 1 and sys.argv[1] == "smoke"))
