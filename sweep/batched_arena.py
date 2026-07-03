"""Thin Python wrapper over the RUST leaf-parallel PUCT arena
(`fastchess.arena_match`). The ENTIRE MCTS — descents, virtual loss, expand,
backup, terminal/threefold, encode — lives in Rust. Python does exactly one
thing per round: the batched GPU forward (`eval_fn`). There is NO Python search
code in this module (the old PNode/LeafPuct tree was removed when the Rust port
landed; see sweep/test_arena_rust.py for the L=1 bit-exact check + reference).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
for _p in (str(ROOT), str(ROOT / "fastchess" / "pybuild")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fastchess  # noqa: E402
from alphazero.chess_net import ChessNet, ChessEvaluator  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_C_PUCT = 1.5
MAX_PLY = 160
MAT_THRESH = 2.0
OPEN_PLIES = 4
DEFAULT_L = 32
OPENINGS_FILE = ROOT / "sweep" / "openings.epd"


def load_opening_suite(path=OPENINGS_FILE):
    """The fixed opening suite (full FENs, one per line). Used by the config
    tournament so every comparison uses controlled, color-balanced starts."""
    return [ln.strip() for ln in open(path) if ln.strip()]


def load_evaluator(path, device=DEV) -> ChessEvaluator:
    ck = torch.load(path, map_location=device)
    net = ChessNet(ck["channels"], ck["blocks"]).to(device).eval()
    net.load_state_dict(ck["state_dict"])
    return ChessEvaluator(net, device=device)


def make_eval_fn(ev: ChessEvaluator, fp16=True):
    """Build the batched GPU-forward callback the Rust arena calls each round:
    planes (M,18,8,8) f32 -> (logits (M,4672) f32, values (M,) f32). This is the
    ONLY Python work per round; the search itself is entirely in Rust."""
    net = ev.net
    use_half = fp16 and ev.device == "cuda"
    if use_half:
        net = net.half()
    net.eval()
    dev = ev.device

    @torch.no_grad()
    def eval_fn(planes):
        x = torch.from_numpy(planes).to(dev)
        if use_half:
            x = x.half()
        logits, values = net(x)
        return (np.ascontiguousarray(logits.float().cpu().numpy(), dtype=np.float32),
                np.ascontiguousarray(values.float().cpu().numpy(), dtype=np.float32))

    return eval_fn


def play_match_timed(path_a, path_b, n_games, ms_per_move, seed=0, *, L=DEFAULT_L,
                     device=DEV, max_ply=MAX_PLY, open_plies=OPEN_PLIES,
                     c_puct=EVAL_C_PUCT, mat_thresh=MAT_THRESH, return_stats=False):
    """Score of A (wins + 0.5*draws) over n_games at a wall-clock ms_per_move
    budget, colors balanced. Routes to fastchess.arena_match (Rust)."""
    evA = load_evaluator(path_a, device)
    evB = load_evaluator(path_b, device)
    efA, efB = make_eval_fn(evA), make_eval_fn(evB)
    score, st = fastchess.arena_match(efA, efB, n_games, float(ms_per_move), 0, 0,
                                      L, L, c_puct, max_ply, mat_thresh, open_plies,
                                      seed, False)
    return (score, st) if return_stats else score


def play_match_fixed(path_a, path_b, n_games, sims, seed=0, *, L=DEFAULT_L,
                     l_a=None, l_b=None, sims_a=None, sims_b=None, device=DEV,
                     max_ply=MAX_PLY, open_plies=OPEN_PLIES, c_puct=EVAL_C_PUCT,
                     mat_thresh=MAT_THRESH, return_stats=False):
    """Fixed-sim match (timer bypassed). l_a/l_b allow different leaf-batch per
    side (e.g. L=1 vs L=k); sims_a/sims_b allow different per-side sim budgets
    (default both = sims). Routes to fastchess.arena_match (Rust)."""
    evA = load_evaluator(path_a, device)
    evB = load_evaluator(path_b, device)
    efA, efB = make_eval_fn(evA), make_eval_fn(evB)
    la = l_a if l_a is not None else L
    lb = l_b if l_b is not None else L
    sa = sims_a if sims_a is not None else sims
    sb = sims_b if sims_b is not None else sims
    score, st = fastchess.arena_match(efA, efB, n_games, 0.0, int(sa), int(sb), la, lb,
                                      c_puct, max_ply, mat_thresh, open_plies,
                                      seed, False)
    return (score, st) if return_stats else score


def play_match_openings(path_a, path_b, sims_a, sims_b, seed=0, *, l_a=DEFAULT_L,
                        l_b=DEFAULT_L, games_per_opening_pair=1, openings=None,
                        device=DEV, max_ply=MAX_PLY, c_puct=EVAL_C_PUCT,
                        mat_thresh=MAT_THRESH, record_moves=False, return_stats=False):
    """Tournament score of A over the FIXED opening suite, with PER-SIDE sims so
    each net is evaluated at its own training-sims operating point. Each opening is
    played BOTH colors (balanced), deterministic best-play from the book position.
    Total games = len(suite) * games_per_opening_pair * 2."""
    fens = openings if openings is not None else load_opening_suite()
    evA = load_evaluator(path_a, device)
    evB = load_evaluator(path_b, device)
    efA, efB = make_eval_fn(evA), make_eval_fn(evB)
    score, st = fastchess.arena_match_openings(
        efA, efB, fens, games_per_opening_pair, 0.0, int(sims_a), int(sims_b),
        l_a, l_b, c_puct, max_ply, mat_thresh, seed, record_moves)
    return (score, st) if return_stats else score
