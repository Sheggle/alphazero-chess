"""Long INDEFINITE training run with the best swept config.

Fresh net (or resume from <run_dir>/latest.pt). Varied per-iter sims (the v2 lever).
Cosine LR from lr0 -> lr1 over a nominal frame horizon, then held constant. Periodic
NAMED checkpoints (iter_NNNNN.pt every CKPT_FRAMES) for the Elo-vs-frames curve later,
plus latest.pt (with optimizer state) every iter for clean resume. Runs until killed.

  PYTHONPATH=.:fastchess/pybuild python sweep/train_long.py <config.json>
  env: NOMINAL_FRAMES=40e6  CKPT_FRAMES=500000
"""
import json, math, sys, time, random, os
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "fastchess" / "pybuild"))
import fastchess
from alphazero.chess_net import ChessNet

ACTION = 4672
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NOMINAL_FRAMES = float(os.environ.get("NOMINAL_FRAMES", 40e6))   # cosine horizon, then hold lr1
CKPT_FRAMES = float(os.environ.get("CKPT_FRAMES", 500_000))      # named-snapshot cadence


def main(cfg_path):
    cfg = json.loads(Path(cfg_path).read_text())
    rdir = Path(cfg["run_dir"]); rdir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.get("seed", 0)); torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    lr0 = float(cfg["lr"]); lr1 = lr0 * float(cfg["lr_final_frac"])
    ch, bl = int(cfg["channels"]), int(cfg["blocks"])

    net = ChessNet(ch, bl).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=lr0, weight_decay=float(cfg.get("weight_decay", 1e-4)))

    frames = 0; it = 0; snap = 0
    latest = rdir / "latest.pt"
    if latest.exists():
        ck = torch.load(latest, map_location=DEV)
        net.load_state_dict(ck["state_dict"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        frames = int(ck.get("frames", 0)); it = int(ck.get("iter", 0)); snap = int(ck.get("snap", 0))
        print(f"RESUMED from {latest.name}: it {it}, {frames/1e6:.2f}M frames, snap {snap}", flush=True)

    @torch.no_grad()
    def eval_fn(planes, lr_, lc_):
        x = torch.from_numpy(planes).to(DEV)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(DEV == "cuda")):
            logits, values = net(x)
        logits = logits.float()
        r = torch.from_numpy(lr_).to(DEV); c = torch.from_numpy(lc_).to(DEV)
        return (np.ascontiguousarray(logits[r, c].cpu().numpy(), dtype=np.float32),
                np.ascontiguousarray(values.float().cpu().numpy(), dtype=np.float32))

    buf = []
    bufmax = int(cfg["buffer_size"]); bs = int(cfg["batch_size"]); tsteps = int(cfg["train_steps"])
    ecoef = float(cfg["entropy_coef"])
    slo = int(cfg["sims_lo"]); shi = max(slo, int(cfg["sims_hi"]))
    nparams = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"LONG run {rdir.name}: {nparams:.1f}M {ch}ch/{bl}b sims={slo}-{shi} ng={cfg['n_games']} "
          f"lr={lr0:.1e}->{lr1:.1e} buf={bufmax} ts={tsteps} | cosine over {NOMINAL_FRAMES/1e6:.0f}M, "
          f"ckpt every {CKPT_FRAMES/1e6:.2f}M", flush=True)

    t0 = time.time()
    while True:
        it += 1
        frac = min(1.0, frames / NOMINAL_FRAMES)
        lr_now = lr1 + 0.5 * (lr0 - lr1) * (1 + math.cos(math.pi * frac))
        for pg in opt.param_groups:
            pg["lr"] = lr_now
        net.eval()
        sims_it = slo if slo == shi else int(round(math.exp(
            np.random.uniform(math.log(slo), math.log(shi)))))
        mc_it = min(int(cfg["mc"]), sims_it)
        samples, _ = fastchess.run_selfplay(
            eval_fn, int(cfg["n_games"]), sims_it, mc_it,
            float(cfg.get("c_visit", 50.0)), float(cfg["c_scale"]), float(cfg.get("c_puct", 1.5)),
            int(cfg.get("max_ply", 120)), float(cfg["mat_thresh"]), True, seed * 100000 + it)
        buf.extend(samples)
        if len(buf) > bufmax:
            del buf[:len(buf) - bufmax]
        frames += len(samples)
        net.train()
        pl = vl = 0.0; nb = 0
        if len(buf) >= bs:
            for _ in range(tsteps):
                idx = np.random.randint(0, len(buf), size=bs)
                batch = [buf[i] for i in idx]
                planes = torch.from_numpy(np.stack([b[0] for b in batch]).astype(np.float32)).to(DEV)
                rows = np.concatenate([np.full(len(b[1]), k, dtype=np.int64) for k, b in enumerate(batch)])
                cols = np.concatenate([b[1].astype(np.int64) for b in batch])
                vals = np.concatenate([b[2] for b in batch]).astype(np.float32)
                tpi = torch.zeros(bs, ACTION, device=DEV)
                tpi[torch.from_numpy(rows).to(DEV), torch.from_numpy(cols).to(DEV)] = torch.from_numpy(vals).to(DEV)
                tz = torch.from_numpy(np.array([b[3] for b in batch], dtype=np.float32)).to(DEV)
                logits, value = net(planes)
                logp = F.log_softmax(logits, dim=1)
                p_loss = -(tpi * logp).sum(1).mean()
                v_loss = F.mse_loss(value, tz)
                ent = -(logp.exp() * logp).sum(1).mean()
                loss = p_loss + v_loss - ecoef * ent
                opt.zero_grad(); loss.backward(); opt.step()
                pl += p_loss.item(); vl += v_loss.item(); nb += 1
        torch.save({"state_dict": net.state_dict(), "opt": opt.state_dict(), "channels": ch,
                    "blocks": bl, "iter": it, "frames": frames, "snap": snap, "cfg": cfg}, latest)
        if frames >= (snap + 1) * CKPT_FRAMES:
            snap += 1
            torch.save({"state_dict": net.state_dict(), "channels": ch, "blocks": bl,
                        "iter": it, "frames": frames, "cfg": cfg}, rdir / f"iter_{snap:05d}.pt")
            print(f"  [ckpt] iter_{snap:05d}.pt @ {frames/1e6:.2f}M frames", flush=True)
        if it % 10 == 0:
            print(f"  it {it} {frames/1e6:.2f}M frames {time.time()-t0:.0f}s lr {lr_now:.1e} "
                  f"ploss {round(pl/nb,4) if nb else None} vloss {round(vl/nb,4) if nb else None} "
                  f"sims{sims_it}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
