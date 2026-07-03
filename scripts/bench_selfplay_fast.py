"""End-to-end self-play wall-clock: ChessGame vs FastChessGame.

Plays real Gumbel-MCTS self-play games (the actual self-play hot path) with a
64-channel / 6-block net at 32 sims, 40-ply cap, and reports ms/move for each
board backend plus the speedup. This is the number that matters for training
throughput (micro-op speedups are in scripts/bench_movegen.py).

Run from the repo root (the .so is auto-located via chess_env_fast):

    PYTHONPATH=. uv run python scripts/bench_selfplay_fast.py
"""

from __future__ import annotations

import time

import numpy as np
import torch

from alphazero.chess_env import ChessGame
from alphazero.chess_env_fast import FastChessGame
from alphazero.chess_net import ChessEvaluator, ChessNet
from alphazero.gumbel import GumbelMCTS


def play_game(evaluator, game_factory, seed, n_sims=32, max_considered=8, max_ply=40,
              add_noise=False):
    """One Gumbel self-play game. Returns (n_moves, position_fens).

    `add_noise=False` makes the game deterministic given (net, seed), so the two
    backends play the IDENTICAL game and the timing is a controlled comparison of
    board-op cost only. (With noise on, the games diverge harmlessly because the
    Gumbel/Dirichlet draws are assigned in legal-move iteration order, which
    differs between python-chess and shakmaty even though the action-index sets
    are identical. Per-move board work is the same either way.)
    """
    rng = np.random.default_rng(seed)
    state = game_factory()
    ucis = []
    moves = 0
    while not state.is_terminal() and moves < max_ply:
        mcts = GumbelMCTS(evaluator, n_sims=n_sims, max_considered=max_considered,
                          rng=rng)
        action, _ = mcts.run(state, add_noise=add_noise)
        # record the move (decode via the python-chess board view) for a sanity
        # check that both backends play the identical game.
        nxt = state.apply(int(action))
        ucis.append(_diff_uci(state.board, nxt.board))
        state = nxt
        moves += 1
    return moves, ucis


def _diff_uci(b0, b1):
    """Best-effort move label from two consecutive board FENs (sanity only)."""
    return b1.fen().split()[0]


def main():
    torch.manual_seed(0)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    net = ChessNet(channels=64, blocks=6)
    net.eval()
    evaluator = ChessEvaluator(net, device="cpu")

    n_games = 4
    seeds = list(range(n_games))

    backends = {
        "python-chess (ChessGame)": ChessGame,
        "fastchess  (FastChessGame)": FastChessGame,
    }

    # warm-up (JIT/caches/BN) so the first timed game isn't penalized.
    for factory in backends.values():
        play_game(evaluator, factory, seed=999, max_ply=4)

    results = {}
    games_played = {}
    for name, factory in backends.items():
        total_time = 0.0
        total_moves = 0
        seqs = []
        for s in seeds:
            t0 = time.perf_counter()
            moves, ucis = play_game(evaluator, factory, seed=s)
            total_time += time.perf_counter() - t0
            total_moves += moves
            seqs.append(ucis)
        results[name] = (total_time, total_moves)
        games_played[name] = seqs

    # sanity: both backends must have played identical games (same seeds).
    cg_seqs = games_played["python-chess (ChessGame)"]
    fg_seqs = games_played["fastchess  (FastChessGame)"]
    identical = cg_seqs == fg_seqs
    print(f"identical game trajectories across backends: {identical}\n")

    print(f"{'backend':<28}{'games':>7}{'moves':>7}{'total s':>10}{'ms/move':>10}")
    print("-" * 62)
    ms = {}
    for name, (t, m) in results.items():
        per = 1000.0 * t / m
        ms[name] = per
        print(f"{name:<28}{n_games:>7}{m:>7}{t:>10.3f}{per:>10.2f}")

    py = ms["python-chess (ChessGame)"]
    fc = ms["fastchess  (FastChessGame)"]
    print("-" * 62)
    print(f"\nend-to-end self-play speedup: {py / fc:.2f}x "
          f"({(1 - fc / py) * 100:.0f}% less wall-time per move)")
    print(f"  ({py:.1f} ms/move -> {fc:.1f} ms/move, 64ch/6b net, 32 sims, CPU)")
    print("\nNote: the neural-net forward pass dominates CPU self-play (~2/3 of "
          "time), which caps the achievable end-to-end speedup; fastchess removes "
          "essentially all of the python-chess board-op cost underneath it.")


if __name__ == "__main__":
    main()
