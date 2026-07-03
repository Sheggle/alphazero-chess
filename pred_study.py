"""How value+policy predictions evolve across training checkpoints, on a fixed
diverse position set. Positions come from current-net self-play (the Rust engine's
samples already carry encoded planes + the outcome label z); material is read off
the planes. Each checkpoint batch-evals all positions (no search). -> JSON.

Run on the box:  PYTHONPATH=.:fastchess/pybuild python pred_study.py
"""
import sys, json, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "fastchess" / "pybuild"))
import fastchess
from alphazero.chess_net import ChessNet

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PVAL = np.array([1, 3, 3, 5, 9], dtype=np.float32)  # P N B R Q (King excluded)
CKPTS = [("iter_00050", 4.6), ("iter_00100", 9.2), ("iter_00150", 14.5), ("iter_00200", 19.8),
         ("iter_00250", 25.2), ("iter_00300", 30.4), ("iter_00350", 35.6), ("iter_00400", 40.8),
         ("iter_00450", 46.0), ("iter_00500", 51.2), ("iter_00550", 56.4), ("iter_00600", 61.5)]
N_POS = 1000


def load_net(path):
    ck = torch.load(ROOT / path, map_location=DEV)
    m = ChessNet(ck["channels"], ck["blocks"]).to(DEV).eval()
    m.load_state_dict(ck["state_dict"]); return m


# --- 1. generate a diverse position set from current-net self-play ---
gen = load_net("models/chess_gpu/latest.pt")


@torch.no_grad()
def eval_fn(planes, lr, lc):
    x = torch.from_numpy(planes).to(DEV)
    with torch.autocast("cuda", dtype=torch.float16):
        logits, values = gen(x)
    logits = logits.float()
    r = torch.from_numpy(lr).to(DEV); c = torch.from_numpy(lc).to(DEV)
    return (np.ascontiguousarray(logits[r, c].cpu().numpy(), dtype=np.float32),
            np.ascontiguousarray(values.float().cpu().numpy(), dtype=np.float32))


print("generating positions via self-play...", flush=True)
samples, _ = fastchess.run_selfplay(eval_fn, 512, 32, 16, 50.0, 0.3, 1.5, 120, 3.0, True, 12345)
rng = np.random.default_rng(0)
sel = rng.choice(len(samples), size=min(N_POS, len(samples)), replace=False)
planes = np.stack([samples[i][0].astype(np.float32) for i in sel])   # (N,18,8,8), stm-canonical
z = np.array([samples[i][3] for i in sel], dtype=np.float32)          # outcome label, stm perspective
mat = (planes[:, 0:5].sum((2, 3)) @ PVAL) - (planes[:, 6:11].sum((2, 3)) @ PVAL)  # material diff, stm
xt = torch.from_numpy(planes).to(DEV)
print(f"{len(z)} positions | outcome mix W/D/L = {(z>0).sum()}/{(z==0).sum()}/{(z<0).sum()} "
      f"| material std {mat.std():.2f}", flush=True)

# --- 2. eval every checkpoint ---
res = []; prev = None
for label, frames in CKPTS:
    t = time.time(); m = load_net(f"models/chess_gpu/{label}.pt")
    with torch.no_grad():
        logits, values = m(xt)
        v = values.float().cpu().numpy()
        logp = torch.log_softmax(logits.float(), dim=1)
        ent = float((-(logp.exp() * logp).sum(1)).mean())
        top = logits.argmax(1).cpu().numpy()
    nz = z != 0
    rec = {"label": label, "frames": frames,
           "val_acc": float((np.sign(v[nz]) == np.sign(z[nz])).mean()),
           "brier": float(np.mean(((v + 1) / 2 - (z + 1) / 2) ** 2)),
           "matcorr": float(np.corrcoef(v, mat)[0, 1]),
           "val_up_piece": float(v[mat >= 3].mean()) if (mat >= 3).any() else None,
           "val_equal": float(v[np.abs(mat) < 1].mean()) if (np.abs(mat) < 1).any() else None,
           "val_down_piece": float(v[mat <= -3].mean()) if (mat <= -3).any() else None,
           "entropy": ent, "val_std": float(v.std())}
    if prev is not None:
        rec["val_drift_rms"] = float(np.sqrt(np.mean((v - prev["v"]) ** 2)))
        rec["top_agree"] = float((top == prev["top"]).mean())
        rec["policy_kl"] = float((prev["logp"].exp() * (prev["logp"] - logp)).sum(1).mean())
    prev = {"v": v, "top": top, "logp": logp}
    res.append(rec)
    print(f"  {label} ({frames:.0f}M): acc {rec['val_acc']:.2f} brier {rec['brier']:.3f} "
          f"matcorr {rec['matcorr']:+.2f} | up-piece {rec['val_up_piece']:+.2f} eq {rec['val_equal']:+.2f} "
          f"dn {rec['val_down_piece']:+.2f} | ent {ent:.2f} [{time.time()-t:.1f}s]", flush=True)

(ROOT / "pred_study_results.json").write_text(json.dumps(res, indent=2))
print("saved pred_study_results.json", flush=True)
