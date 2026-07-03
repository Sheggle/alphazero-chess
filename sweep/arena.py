"""Eval matches between trained config-nets, at a FIXED search budget (fair
regardless of each config's training sims), routed to the RUST leaf-parallel
PUCT arena (`fastchess.arena_match`). The old batch-1 GumbelMCTS-over-python-chess
`play_match` was REMOVED in the Rust port: the entire search now lives in Rust;
Python only does the batched GPU forward. Bradley-Terry Elo helper retained.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
for _p in (str(ROOT), str(ROOT / "fastchess" / "pybuild")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fastchess  # noqa: E402
from alphazero.chess_net import ChessNet  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_SIMS = 32      # fixed for all configs -> fair
EVAL_L = 8          # leaf-batch (rounds = sims/L); L=8 is fast and ~as strong as L=1 here
MAX_PLY = 160
OPEN_PLIES = 4
C_PUCT = 1.5
MAT_THRESH = 2.0


def _fuse_eval_net(net):
    """BN-fold (eval-exact) + channels_last — the self-play forward wins, applied
    to the play/eval path (they were never carried over here). Folds each BN into
    its bias-free conv and switches to NHWC so cuDNN uses tensor-core kernels with
    no NCHW<->NHWC transposes. Measured forward speedup vs the old fp16-NCHW path:
    ~1.6x at the small batches production uses (one game, leaf-parallel L~=16),
    tapering to ~1.17x at M=1024. Output is bit-close (policy argmax agree 99.6%,
    value MAE 0.003 vs fp32) — same fp16 the eval path already used."""
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


def _eval_fn_for(path, device):
    ck = torch.load(path, map_location=device)
    net = ChessNet(ck["channels"], ck["blocks"]).to(device).eval()
    net.load_state_dict(ck["state_dict"])
    use_half = device == "cuda"
    if use_half:
        net = _fuse_eval_net(net).half()  # BN-fold + channels_last (~1.6x @ prod batch)

    @torch.no_grad()
    def eval_fn(planes):  # (M,18,8,8) f32 -> (logits (M,4672) f32, values (M,) f32)
        x = torch.from_numpy(planes).to(device)
        if use_half:
            x = x.half().contiguous(memory_format=torch.channels_last)
        logits, values = net(x)
        return (np.ascontiguousarray(logits.float().cpu().numpy(), dtype=np.float32),
                np.ascontiguousarray(values.float().cpu().numpy(), dtype=np.float32))

    return eval_fn


def play_match(path_a, path_b, n_games, seed=0):
    """Score of A (wins + 0.5*draws) over n_games, colors balanced, at a fixed
    eval budget (EVAL_SIMS sims, leaf-batch EVAL_L) via the Rust PUCT arena."""
    efA = _eval_fn_for(path_a, DEV)
    efB = _eval_fn_for(path_b, DEV)
    score, _ = fastchess.arena_match(efA, efB, n_games, 0.0, EVAL_SIMS, EVAL_SIMS,
                                     EVAL_L, EVAL_L, C_PUCT, MAX_PLY, MAT_THRESH,
                                     OPEN_PLIES, seed, False)
    return score


def fit_elo(n, W, N):
    """Bradley-Terry MM. W[i,j]=score of i vs j, N=games. Returns centered Elo."""
    gamma = np.ones(n); wins = W.sum(1)
    for _ in range(3000):
        ng = gamma.copy()
        for i in range(n):
            den = sum(N[i, j] / (gamma[i] + gamma[j]) for j in range(n) if j != i and N[i, j] > 0)
            if den > 0 and wins[i] > 0:
                ng[i] = wins[i] / den
        ng = np.clip(ng, 1e-12, None); ng /= np.exp(np.mean(np.log(ng)))
        if np.max(np.abs(ng - gamma)) < 1e-10:
            gamma = ng; break
        gamma = ng
    R = 400 * np.log10(gamma); return R - R.mean()
