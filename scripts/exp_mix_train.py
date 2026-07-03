"""Train one model for the mix experiment. Args: <tag> <mix_frac> <seed>."""
import sys
import torch
torch.set_num_threads(1)
from pathlib import Path
from alphazero.train import robust_lowsim_config, train

tag = sys.argv[1]; mix_frac = float(sys.argv[2]); seed = int(sys.argv[3])
cfg = robust_lowsim_config(iterations=50)
cfg.seed = seed
cfg.mix_frac = mix_frac
cfg.mix_sims = 10
cfg.mix_max_considered = 8
net, ev, cfg = train(cfg, verbose=False)
Path("models").mkdir(exist_ok=True)
torch.save({"state_dict": net.state_dict(), "channels": cfg.channels},
           f"models/ttt_mix_{tag}_s{seed}.pt")
curve = [r["opt_rate_mcts"] for r in cfg.log]
print(f"[{tag} s{seed}] mix={mix_frac} peak={max(curve):.4f} "
      f"last10_min={min(curve[-10:]):.4f} final={curve[-1]:.4f}")
