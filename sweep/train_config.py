"""Train ONE hyperparameter config: fresh net, fixed WALL-TIME budget, cosine LR.

Saves periodic + final checkpoints and a metrics.json to the run dir. No
in-training eval (the sweep evaluates strength separately, by playing configs
against each other). Reads config JSON from argv[1].

  PYTHONPATH=.:fastchess/pybuild python sweep/train_config.py <config.json>
"""
import json, math, sys, time, random
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "fastchess" / "pybuild"))
import fastchess
from alphazero.chess_net import ChessNet

ACTION = 4672
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main(cfg_path):
    cfg = json.loads(Path(cfg_path).read_text())
    rdir = Path(cfg["run_dir"]); rdir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.get("seed", 0)); torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    budget = float(cfg["time_budget_s"])
    lr0 = float(cfg["lr"]); lr1 = lr0 * float(cfg["lr_final_frac"])

    net = ChessNet(int(cfg["channels"]), int(cfg["blocks"])).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=lr0, weight_decay=float(cfg.get("weight_decay", 1e-4)))

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
    nparams = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"cfg {rdir.name}: {nparams:.1f}M {cfg['channels']}ch/{cfg['blocks']}b sims={cfg['sims_lo']}-{cfg['sims_hi']} "
          f"ng={cfg['n_games']} lr={lr0:.1e}->{lr1:.1e} buf={bufmax} ts={tsteps} budget={budget:.0f}s", flush=True)

    t0 = time.time(); it = 0; frames = 0; hist = []
    while time.time() - t0 < budget:
        it += 1
        frac = min(1.0, (time.time() - t0) / budget)
        lr_now = lr1 + 0.5 * (lr0 - lr1) * (1 + math.cos(math.pi * frac))
        for pg in opt.param_groups:
            pg["lr"] = lr_now
        net.eval()
        # varied training sims per iter (log-uniform in [sims_lo, sims_hi]) so the value
        # head sees a range of search depths -> robust at production think-time, not OOD.
        slo = int(cfg["sims_lo"]); shi = max(slo, int(cfg["sims_hi"]))
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
        rec = {"iter": it, "frames": frames, "t": round(time.time() - t0, 1), "lr": lr_now,
               "ploss": round(pl / nb, 4) if nb else None, "vloss": round(vl / nb, 4) if nb else None}
        hist.append(rec)
        torch.save({"state_dict": net.state_dict(), "channels": int(cfg["channels"]),
                    "blocks": int(cfg["blocks"]), "iter": it, "frames": frames, "cfg": cfg},
                   rdir / "final.pt")  # overwrite; final.pt always = latest
        if it % 10 == 0:
            print(f"  it {it} {frames/1e6:.1f}M frames {rec['t']:.0f}s lr {lr_now:.1e} "
                  f"ploss {rec['ploss']} vloss {rec['vloss']}", flush=True)
    (rdir / "metrics.json").write_text(json.dumps({"cfg": cfg, "iters": it, "frames": frames, "hist": hist}))
    print(f"DONE {rdir.name}: {it} iters, {frames/1e6:.2f}M frames in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
