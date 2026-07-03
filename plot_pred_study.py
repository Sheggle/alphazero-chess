"""Plot how value+policy predictions evolve across checkpoints (pred_study_results.json)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
r = json.loads((ROOT / "pred_study_results.json").read_text())
f = np.array([x["frames"] for x in r])

fig, ax = plt.subplots(2, 2, figsize=(13, 9))

# value calibration: accuracy + brier
a = ax[0, 0]
a.plot(f, [x["val_acc"] for x in r], "o-", color="#2266cc", label="sign accuracy vs outcome")
a.set_ylabel("value sign-accuracy"); a.set_ylim(0.4, 1.0); a.grid(alpha=.3); a.legend(loc="upper left")
a2 = a.twinx(); a2.plot(f, [x["brier"] for x in r], "s--", color="#cc6622", label="Brier (lower=better)")
a2.set_ylabel("Brier", color="#cc6622"); a2.legend(loc="lower right")
a.set_title("Value calibration vs game outcome")

# material awareness: corr + separation
a = ax[0, 1]
a.plot(f, [x["matcorr"] for x in r], "o-", color="#119944", label="corr(value, material)")
a.set_ylabel("value↔material corr"); a.grid(alpha=.3); a.set_title("Material awareness of the value")
a.plot(f, [x["val_up_piece"] for x in r], "^-", color="#2266cc", label="mean val | up a piece (+3)")
a.plot(f, [x["val_equal"] for x in r], "s-", color="#888888", label="mean val | equal")
a.plot(f, [x["val_down_piece"] for x in r], "v-", color="#cc2222", label="mean val | down a piece (-3)")
a.legend(fontsize=8); a.set_xlabel("frames (M)")

# policy: entropy + churn
a = ax[1, 0]
a.plot(f, [x["entropy"] for x in r], "o-", color="#7733aa", label="policy entropy")
a.set_ylabel("policy entropy (nats)"); a.grid(alpha=.3); a.set_title("Policy sharpness + churn"); a.set_xlabel("frames (M)")
a2 = a.twinx()
a2.plot(f[1:], [x.get("top_agree") for x in r[1:]], "s--", color="#cc6622", label="top-move agree w/ prev ckpt")
a2.set_ylabel("top-move agreement", color="#cc6622"); a2.set_ylim(0, 1)
a.legend(loc="upper right", fontsize=8); a2.legend(loc="lower right", fontsize=8)

# cross-checkpoint change
a = ax[1, 1]
a.plot(f[1:], [x.get("val_drift_rms") for x in r[1:]], "o-", color="#2266cc", label="value RMS change vs prev")
a.set_ylabel("value drift (RMS)"); a.grid(alpha=.3); a.set_title("How much predictions change per checkpoint"); a.set_xlabel("frames (M)")
a2 = a.twinx()
a2.plot(f[1:], [x.get("policy_kl") for x in r[1:]], "s--", color="#cc6622", label="policy KL vs prev")
a2.set_ylabel("policy KL", color="#cc6622")
a.legend(loc="upper right", fontsize=8); a2.legend(loc="lower right", fontsize=8)

fig.suptitle("Value/Policy evolution across training (1000 fixed positions)", fontsize=13)
fig.tight_layout()
out = ROOT / "pred_study.png"
fig.savefig(out, dpi=130)
print(f"saved {out}")
