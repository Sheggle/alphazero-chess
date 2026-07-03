"""Validation harness for the Rust batched self-play engine (fastchess.run_selfplay).

Proves the Rust engine reproduces the existing Python search EXACTLY:

  * Core (add_noise=False): for a fixed net + seed, the Rust engine's per-move
    chosen action and completed-Q policy target must match the reference
    `GumbelMCTS.run` / `play_chess_game` loop (same net, same FastChessGame board
    + encode + legal order). This is the search-correctness proof — any mismatch
    here would silently corrupt training. We report per-move mismatch counts and
    max |pi| diff. The only expected residual is ~1e-6 from f32 softmax/exp
    library differences (numpy vs Rust), exactly analogous to the documented
    batch-vs-single float noise; the argmax action is robust to it.

  * Gumbel (add_noise=True): can't be bit-identical to numpy's PCG64 Gumbel
    stream, but it's iid Gumbel(0,1) either way (distributionally equivalent, so
    elo-vs-frames is preserved). We check legality, well-formedness, value
    targets, and exact reproducibility (same seed -> same output).

Usage:  python validate_selfplay.py [--device cpu|cuda]
"""
from __future__ import annotations

import sys

import numpy as np
import torch

# Load the Rust extension from the build dir (avoids the namespace-package shadow
# documented in chess_env_fast).
import pathlib
_so = str(pathlib.Path(__file__).resolve().parent / "fastchess" / "pybuild")
if _so not in sys.path:
    sys.path.insert(0, _so)
import fastchess  # noqa: E402
assert hasattr(fastchess, "run_selfplay"), "rebuild fastchess (run_selfplay missing)"

from alphazero.chess_env import ACTION_SIZE  # noqa: E402
from alphazero.chess_env_fast import FastChessGame  # noqa: E402
from alphazero.chess_encode import encode_state  # noqa: E402
from alphazero.chess_net import ChessEvaluator, ChessNet  # noqa: E402
from alphazero.gumbel import GumbelMCTS  # noqa: E402

C_PUCT = 1.5


def _outcome_white(g, mat_thresh):
    if g.is_terminal():
        return g.result()
    d = g.material_diff()
    return 1 if d >= mat_thresh else (-1 if d <= -mat_thresh else 0)


def reference_game(ev, sims, mc, max_ply, c_visit, c_scale, mat_thresh):
    """Mirror of chess_train.play_chess_game, but add_noise=False (deterministic)
    so it is exactly reproducible by the Rust engine. Uses FastChessGame so the
    encode + legal-move order are identical to Rust."""
    rng = np.random.default_rng(0)  # unused for add_noise=False, but required
    g = FastChessGame()
    recs, moves = [], []
    while not g.is_terminal() and g.ply < max_ply:
        a, pi = GumbelMCTS(ev, n_sims=sims, max_considered=mc, c_visit=c_visit,
                           c_scale=c_scale, c_puct=C_PUCT, rng=rng).run(g, add_noise=False)
        recs.append((pi, g.to_play, g.material_diff()))
        moves.append(int(a))
        g = g.apply(int(a))
    terminal = g.is_terminal()
    z_white = _outcome_white(g, mat_thresh)
    samples = []
    for pi, to_play, mat_w in recs:
        # Binary "just who wins" value target, matching the engine's current
        # finalize (z = z_white * to_play; z_white = terminal result, or the
        # material-adjudicated winner for capped games). The old tanh material-
        # graded target was retired when the engine switched to binary z.
        z = float(z_white * to_play)
        idx = np.nonzero(pi)[0].astype(np.int16)
        samples.append((idx, pi[idx].astype(np.float32), np.float32(z)))
    stats = {"terminal": terminal, "plies": g.ply, "z_white": z_white,
             "result": g.result() if terminal else None}
    return moves, samples, stats


def make_eval_fn(net, device):
    @torch.no_grad()
    def eval_fn(planes, legal_rows, legal_cols):  # planes (B,18,8,8) f32; legal_* (M,) int64
        net.eval()
        x = torch.from_numpy(planes).to(device)
        logits, values = net(x)
        # Gather ONLY the legal logits on-device, in the order Rust sent them, so
        # only ~B*35 floats cross back (no full (B,4672) D2H). Same f32 values as
        # indexing the full row -> bit-identical priors after Rust's softmax.
        r = torch.from_numpy(legal_rows).to(device)
        c = torch.from_numpy(legal_cols).to(device)
        legal_logits = np.ascontiguousarray(
            logits[r, c].detach().float().cpu().numpy(), dtype=np.float32)
        values = np.ascontiguousarray(values.detach().float().cpu().numpy(), dtype=np.float32)
        return legal_logits, values
    return eval_fn


def _full_pi(idx, val):
    pi = np.zeros(ACTION_SIZE, dtype=np.float64)
    pi[np.asarray(idx, dtype=np.int64)] = val
    return pi


def validate_exact(device, settings):
    net = ChessNet(channels=16, blocks=2)
    net.eval()
    ev = ChessEvaluator(net, device=device)
    net.to(device)
    eval_fn = make_eval_fn(net, device)

    total_moves = total_pi = 0
    action_mismatch = pi_close_fail = z_mismatch = 0
    max_pi_diff = 0.0
    n_games_compared = 0

    for (sims, mc, max_ply) in settings:
        mat_thresh = 1.0
        c_visit, c_scale = 50.0, 1.0
        seed = 12345
        # Reference (single-game GumbelMCTS) and Rust (n_games=1) from start pos.
        ref_moves, ref_samples, ref_stats = reference_game(
            ev, sims, mc, max_ply, c_visit, c_scale, mat_thresh)
        rust_samples, rust_stats = fastchess.run_selfplay(
            eval_fn, 1, sims, mc, c_visit, c_scale, C_PUCT, max_ply, mat_thresh,
            False, seed)
        st = rust_stats[0]
        rust_moves = list(st["moves"])
        n_games_compared += 1

        # Move sequence.
        ml = min(len(ref_moves), len(rust_moves))
        if ref_moves != rust_moves:
            action_mismatch += sum(1 for i in range(ml) if ref_moves[i] != rust_moves[i])
            action_mismatch += abs(len(ref_moves) - len(rust_moves))
        total_moves += max(len(ref_moves), len(rust_moves))

        # Per-move pi + z.
        nl = min(len(ref_samples), len(rust_samples))
        for i in range(nl):
            r_idx, r_val, r_z = ref_samples[i]
            x_planes, x_idx, x_val, x_z = rust_samples[i]
            pa = _full_pi(r_idx, r_val)
            pb = _full_pi(x_idx, x_val)
            d = float(np.max(np.abs(pa - pb)))
            max_pi_diff = max(max_pi_diff, d)
            if not np.allclose(pa, pb, atol=2e-5, rtol=0):
                pi_close_fail += 1
            if abs(float(r_z) - float(x_z)) > 1e-6:
                z_mismatch += 1
            total_pi += 1

        # Stats sanity.
        assert st["terminal"] == ref_stats["terminal"], (st["terminal"], ref_stats)
        assert st["z_white"] == ref_stats["z_white"], (st["z_white"], ref_stats)
        assert st["plies"] == ref_stats["plies"], (st["plies"], ref_stats)
        print(f"  [{sims=} {mc=} {max_ply=}] moves ref={len(ref_moves)} rust={len(rust_moves)} "
              f"terminal={st['terminal']} z_white={st['z_white']}")

    print(f"\n=== EXACT (add_noise=False) over {n_games_compared} games ===")
    print(f"moves compared       : {total_moves}")
    print(f"action mismatches    : {action_mismatch}")
    print(f"policy targets        : {total_pi}")
    print(f"pi allclose failures : {pi_close_fail} (atol=2e-5)")
    print(f"max |pi| diff        : {max_pi_diff:.2e}")
    print(f"z mismatches         : {z_mismatch}")
    return action_mismatch, pi_close_fail, z_mismatch, max_pi_diff


def validate_noise(device):
    net = ChessNet(channels=16, blocks=2)
    net.eval()
    net.to(device)
    eval_fn = make_eval_fn(net, device)
    kw = dict(sims=16, max_considered=8)

    # Reproducibility: same seed -> identical output.
    s1, st1 = fastchess.run_selfplay(eval_fn, 8, 16, 8, 50.0, 1.0, C_PUCT, 30, 1.0, True, 777)
    s2, st2 = fastchess.run_selfplay(eval_fn, 8, 16, 8, 50.0, 1.0, C_PUCT, 30, 1.0, True, 777)
    repro = (len(s1) == len(s2))
    for (p1, i1, v1, z1), (p2, i2, v2, z2) in zip(s1, s2):
        if not (np.array_equal(i1, i2) and np.array_equal(v1, v2) and z1 == z2):
            repro = False
            break
    # Well-formedness + legality of support.
    bad = 0
    for planes, idx, vals, z in s1:
        if planes.shape != (18, 8, 8) or planes.dtype != np.float16:
            bad += 1
        if idx.dtype != np.int16 or vals.dtype != np.float32:
            bad += 1
        if len(idx) != len(vals) or len(idx) == 0:
            bad += 1
        if abs(float(vals.sum()) - 1.0) > 1e-4:
            bad += 1
        if float(z) < -1.0001 or float(z) > 1.0001:
            bad += 1
    print(f"\n=== Gumbel (add_noise=True) ===")
    print(f"reproducible (same seed)   : {repro}")
    print(f"games                       : {len(st1)}  samples: {len(s1)}")
    print(f"malformed-sample checks     : {bad}")
    return repro, bad


def main():
    device = "cpu"
    if "--device" in sys.argv:
        device = sys.argv[sys.argv.index("--device") + 1]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    torch.manual_seed(0)
    print(f"device = {device}")
    settings = [(2, 2, 10), (4, 4, 12), (8, 4, 16), (8, 8, 20), (16, 8, 24),
                (16, 16, 30), (24, 8, 40), (32, 8, 30), (32, 16, 50), (48, 16, 60)]
    am, pf, zm, mpd = validate_exact(device, settings)
    repro, bad = validate_noise(device)

    ok = (am == 0 and pf == 0 and zm == 0 and repro and bad == 0)
    print("\n================ RESULT ================")
    print("PASS" if ok else "FAIL",
          f"| action_mismatch={am} pi_fail={pf} z_mismatch={zm} "
          f"max_pi_diff={mpd:.2e} repro={repro} malformed={bad}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
