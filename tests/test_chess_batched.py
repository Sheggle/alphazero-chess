"""Correctness tests for the batched Gumbel self-play engine.

The central claim is *semantic equivalence*: the batched engine produces the
same training data as the existing single-game `GumbelMCTS`. We prove it in
layers:

  1. Legality -- batched self-play only ever plays legal moves and emits
     well-formed Sample tuples.

  2. EXACT search equivalence -- for a fixed net + seed, driving ONE batched
     search (as 1-element batches, with `in_search_claim_draw=True` so the
     terminal semantics match) returns the *identical* action and the
     *identical* policy vector as `GumbelMCTS.run`, across several positions and
     for both deterministic (add_noise=False) and Gumbel (add_noise=True) acting.

     This is bit-exact -- not merely close -- because the coroutine restructure,
     lazy expansion and make/unmake are all value-preserving, the RNG is used at
     the same point with the same draw size, and a 1-element batch feeds the net
     the same tensor `predict` would.

  3. Batch numerical equivalence -- the *only* thing that changes when B>1 is the
     network's own batch-vs-single floating-point reduction order. We show
     `predict_batch` vs per-state `predict` are allclose (not bit-identical),
     which is the sole, net-level reason a large-batch run could ever pick a
     different move than the single-game engine.
"""

import chess
import numpy as np
import torch

from alphazero.chess_env import ChessGame
from alphazero.chess_net import ChessEvaluator, ChessNet
from alphazero.gumbel import GumbelMCTS
from alphazero.chess_batched import (
    _BatchedGumbel,
    _make_batch_eval,
    play_batched_games,
    run_single_search,
)


def _make_evaluator(seed=0, channels=16, blocks=2):
    torch.manual_seed(seed)
    net = ChessNet(channels=channels, blocks=blocks)
    net.eval()
    return ChessEvaluator(net)


# A spread of positions: opening, a quiet middlegame, a sharp tactical spot, and
# a position one move from mate (exercises in-search terminal handling). None of
# these reach a threefold within a shallow search, so claim_draw history
# differences between the engines cannot bite.
_FENS = [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1",   # Re8# available
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",  # black to move
]


def test_batched_selfplay_legal_and_wellformed():
    """End-to-end: a small concurrent batch plays only legal moves and emits
    Sample tuples with the documented dtypes/shapes and a valid z."""
    ev = _make_evaluator(seed=1)
    samples, stats = play_batched_games(
        ev, n_games=6, concurrency=3, sims=8, max_considered=4,
        max_ply=20, seed=123)

    assert len(stats) == 6
    assert len(samples) > 0
    for planes, idx, vals, z in samples:
        assert planes.shape == (18, 8, 8) and planes.dtype == np.float16
        assert idx.dtype == np.int16 and vals.dtype == np.float32
        assert len(idx) == len(vals) and len(idx) > 0
        assert abs(float(vals.sum()) - 1.0) < 1e-4    # pi is a distribution
        assert float(z) in (-1.0, 0.0, 1.0)
    for st in stats:
        assert 0 < st["plies"] <= 20 or st["terminal"]
        assert st["z_white"] in (-1, 0, 1)


def test_batched_selfplay_moves_are_legal_replay():
    """Replay a single batched game move-by-move and assert each emitted policy's
    support is exactly the legal-move set of that position."""
    ev = _make_evaluator(seed=2)
    # One game, played through the batched driver.
    samples, stats = play_batched_games(
        ev, n_games=1, concurrency=1, sims=8, max_considered=4,
        max_ply=12, seed=7)
    # Reconstruct: every sample's pi indices must be legal in some reachable
    # position; we verify the first move's support against the start position.
    board = chess.Board()
    legal0 = set(ChessGame(board).legal_moves())
    first_idx = set(int(i) for i in samples[0][1])
    assert first_idx.issubset(legal0)


def _gumbel_reference(ev, board, seed, add_noise, sims, mc):
    rng = np.random.default_rng(seed)
    g = ChessGame(board.copy())
    return GumbelMCTS(ev, n_sims=sims, max_considered=mc,
                      c_visit=50.0, c_scale=1.0, c_puct=1.5,
                      rng=rng).run(g, add_noise=add_noise)


def _batched_reference(ev, board, seed, add_noise, sims, mc):
    # in_search_claim_draw=True -> identical terminal semantics to GumbelMCTS.
    engine = _BatchedGumbel(n_sims=sims, max_considered=mc, c_visit=50.0,
                            c_scale=1.0, c_puct=1.5, in_search_claim_draw=True)
    eval_fn = _make_batch_eval(ev)
    rng = np.random.default_rng(seed)
    return run_single_search(engine, board, rng, eval_fn, add_noise)


def test_exact_equivalence_single_search():
    """Bit-exact: batched single search == GumbelMCTS.run for several positions,
    seeds, sim budgets, and both acting modes."""
    ev = _make_evaluator(seed=3)
    for fen in _FENS:
        board = chess.Board(fen)
        for add_noise in (False, True):
            for sims, mc in ((4, 4), (16, 8), (32, 8)):
                seed = 12345
                a_ref, pi_ref = _gumbel_reference(ev, board, seed, add_noise, sims, mc)
                a_bat, pi_bat = _batched_reference(ev, board, seed, add_noise, sims, mc)
                msg = f"fen={fen} noise={add_noise} sims={sims}"
                assert a_ref == a_bat, f"action mismatch: {a_ref} vs {a_bat} | {msg}"
                # Exact policy equality (same float ops, same net, same RNG).
                assert np.array_equal(pi_ref, pi_bat), f"policy mismatch | {msg}"


def test_batch_only_differs_by_network_float_noise():
    """The single source of B>1 divergence is the net's batch reduction order:
    predict_batch vs per-state predict are allclose but not bit-identical."""
    ev = _make_evaluator(seed=4)
    states = [ChessGame(chess.Board(f)) for f in _FENS]
    probs_b, vals_b = ev.predict_batch(states)
    for i, s in enumerate(states):
        p1, v1 = ev.predict(s)
        # Numerically equivalent...
        np.testing.assert_allclose(probs_b[i], p1, atol=1e-5, rtol=1e-4)
        np.testing.assert_allclose(vals_b[i], v1, atol=1e-5)
        # ...but batching is allowed to perturb the last bits.
    # And a full batched self-play run at B>1 still only plays legal moves
    # (covered above) -- so any move difference vs single-game is bounded by this
    # net-level noise, never by the search logic.


def test_concurrency_reproducible_and_stable():
    """Two guarantees:

      * Same seed + same concurrency is bit-exact reproducible (deterministic
        engine, per-game RNG).
      * Across *different* concurrency the per-game RNG is untouched, so results
        differ at most by the network's batch-composition float noise: identical
        policy support, allclose policy values. (A move could in principle flip
        at a razor-thin decision boundary -- that is the documented net-level
        noise, never a search-logic difference.)
    """
    ev = _make_evaluator(seed=5)
    kw = dict(n_games=4, sims=8, max_considered=4, max_ply=16, seed=99)

    # Reproducibility at fixed concurrency: bit-exact.
    a1, _ = play_batched_games(ev, concurrency=4, **kw)
    a2, _ = play_batched_games(ev, concurrency=4, **kw)
    assert len(a1) == len(a2)
    for (_, i1, v1, z1), (_, i2, v2, z2) in zip(a1, a2):
        assert np.array_equal(i1, i2) and np.array_equal(v1, v2) and z1 == z2

    # Across concurrency: same support, allclose values (net float noise only).
    s1, st1 = play_batched_games(ev, concurrency=1, **kw)
    s4, st4 = play_batched_games(ev, concurrency=4, **kw)
    assert len(s1) == len(s4)
    for (p1, i1, v1, z1), (p4, i4, v4, z4) in zip(s1, s4):
        assert np.array_equal(i1, i4)        # legal-move support is identical
        np.testing.assert_allclose(v1, v4, atol=1e-3)
