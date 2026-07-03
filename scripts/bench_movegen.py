"""Benchmark + correctness harness: fastchess (Rust/shakmaty via pyo3) vs python-chess.

Run from the research repo root:

    PYTHONPATH=.:fastchess/pybuild uv run python scripts/bench_movegen.py

Covers the AlphaZero self-play hot path (see alphazero/chess_env.py):
  (a) legal move generation
  (b) make-move (board.copy(stack=False)+push  vs  fastchess clone+apply)
  (c) terminal + result detection (is_game_over(claim_draw=True) / outcome)

Also verifies that fastchess's legal move sets, terminal flags and results
exactly match python-chess across many random positions (incl. forced
repetition / fifty-move lines), then estimates end-to-end self-play speedup.
"""

from __future__ import annotations

import random
import time

import chess

import fastchess


# --------------------------------------------------------------------------
# Position generation
# --------------------------------------------------------------------------
def random_games(n_games: int, max_plies: int, seed: int):
    """Play random games; yield (move_uci_history, terminal?) per game.

    Replaying the uci history on both engines from the start gives matching
    repetition / fifty-move history (needed for threefold parity).
    """
    rng = random.Random(seed)
    games = []
    for _ in range(n_games):
        board = chess.Board()
        history = []
        for _ in range(max_plies):
            if board.is_game_over(claim_draw=True):
                break
            mv = rng.choice(list(board.legal_moves))
            history.append(mv.uci())
            board.push(mv)
        games.append(history)
    return games


def repetition_lines():
    """Deterministic lines that force threefold / fivefold repetition."""
    # Shuffle knights back and forth: Nf3 Nf6 Ng1 Ng8 ... repeats start position.
    base = ["g1f3", "g8f6", "f3g1", "f6g8"]
    return [base * 1, base * 2, base * 3, base * 4]  # 1..4 cycles


# --------------------------------------------------------------------------
# Correctness
# --------------------------------------------------------------------------
def py_result(board: chess.Board) -> int:
    oc = board.outcome(claim_draw=True)
    if oc is None or oc.winner is None:
        return 0
    return 1 if oc.winner == chess.WHITE else -1


def check_correctness(games):
    moves_checked = 0
    legal_mismatch = 0
    terminal_mismatch = 0
    result_mismatch = 0
    examples = []

    for history in games:
        board = chess.Board()
        fb = fastchess.Board()
        # check the start position too
        steps = [None] + list(history)
        for mv in steps:
            if mv is not None:
                board.push_uci(mv)
                fb.apply_uci(mv)

            moves_checked += 1

            py_legal = {m.uci() for m in board.legal_moves}
            fc_legal = set(fb.legal_uci())
            if py_legal != fc_legal:
                legal_mismatch += 1
                if len(examples) < 5:
                    examples.append(
                        ("legal", board.fen(), py_legal ^ fc_legal)
                    )

            py_term = board.is_game_over(claim_draw=True)
            fc_term = fb.is_terminal()
            if py_term != fc_term:
                terminal_mismatch += 1
                if len(examples) < 5:
                    examples.append(("terminal", board.fen(), (py_term, fc_term)))

            pr, fr = py_result(board), fb.result()
            if pr != fr:
                result_mismatch += 1
                if len(examples) < 5:
                    examples.append(("result", board.fen(), (pr, fr)))

            if py_term:
                break

    return {
        "moves_checked": moves_checked,
        "legal_mismatch": legal_mismatch,
        "terminal_mismatch": terminal_mismatch,
        "result_mismatch": result_mismatch,
        "examples": examples,
    }


# --------------------------------------------------------------------------
# Benchmark
# --------------------------------------------------------------------------
def collect_positions(games, cap):
    """Sample non-terminal positions as (fen, history_uci) for both engines."""
    samples = []
    for history in games:
        board = chess.Board()
        partial = []
        for mv in [None] + list(history):
            if mv is not None:
                board.push_uci(mv)
                partial.append(mv)
            if board.is_game_over(claim_draw=True):
                break
            if list(board.legal_moves):
                samples.append(list(partial))
        if len(samples) >= cap:
            break
    return samples[:cap]


def build_boards(samples):
    py_boards, fc_boards = [], []
    for hist in samples:
        b = chess.Board()
        fb = fastchess.Board()
        for mv in hist:
            b.push_uci(mv)
            fb.apply_uci(mv)
        py_boards.append(b)
        fc_boards.append(fb)
    return py_boards, fc_boards


def timeit(fn, repeats):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        n = fn()
        dt = time.perf_counter() - t0
        best = min(best, dt / n)
    return 1.0 / best  # ops/sec


def run_benchmarks(py_boards, fc_boards, repeats=5):
    npos = len(py_boards)
    results = {}

    # (a) legal move generation
    def py_legal():
        for b in py_boards:
            list(b.legal_moves)
        return npos

    def fc_legal():
        for b in fc_boards:
            b.legal_tuples()
        return npos

    results["legal_gen"] = (timeit(py_legal, repeats), timeit(fc_legal, repeats))

    # (b) make-move: copy(stack=False)+push  vs  clone+apply_index(0)
    #     (mirrors ChessGame.apply, which builds a fresh game each ply)
    py_first = [next(iter(b.legal_moves)) for b in py_boards]

    def py_make():
        for b, mv in zip(py_boards, py_first):
            nb = b.copy(stack=False)
            nb.push(mv)
        return npos

    def fc_make():
        for b in fc_boards:
            b.apply_index_copy(0)
        return npos

    results["make_copy"] = (timeit(py_make, repeats), timeit(fc_make, repeats))

    # make/unmake (push/pop) — python-chess's cheapest successor, no Rust analog
    # other than clone+apply; reported for reference.
    def py_pushpop():
        for b, mv in zip(py_boards, py_first):
            b.push(mv)
            b.pop()
        return npos

    results["make_pushpop_vs_clone"] = (
        timeit(py_pushpop, repeats),
        timeit(fc_make, repeats),
    )

    # (c) terminal + result
    def py_term():
        for b in py_boards:
            if b.is_game_over(claim_draw=True):
                py_result(b)
        return npos

    def fc_term():
        for b in fc_boards:
            if b.is_terminal():
                b.result()
        return npos

    results["terminal_result"] = (timeit(py_term, repeats), timeit(fc_term, repeats))

    return results


# --------------------------------------------------------------------------
def main():
    print("Generating random games...")
    games = random_games(n_games=400, max_plies=160, seed=12345)
    games += repetition_lines()

    print("Checking correctness (legal sets / terminal / result)...")
    corr = check_correctness(games)
    print(f"  positions checked : {corr['moves_checked']}")
    print(f"  legal mismatches  : {corr['legal_mismatch']}")
    print(f"  terminal mismatch : {corr['terminal_mismatch']}")
    print(f"  result mismatch   : {corr['result_mismatch']}")
    for kind, fen, detail in corr["examples"]:
        print(f"    [{kind}] {fen}  ->  {detail}")

    print("\nCollecting benchmark positions...")
    samples = collect_positions(games, cap=8000)
    py_boards, fc_boards = build_boards(samples)
    print(f"  {len(py_boards)} positions")

    print("Benchmarking (best of 5 passes each)...\n")
    res = run_benchmarks(py_boards, fc_boards, repeats=5)

    print(f"{'operation':<26}{'python-chess':>16}{'fastchess':>16}{'speedup':>10}")
    print("-" * 68)
    speedups = {}
    for name, (py_ops, fc_ops) in res.items():
        sp = fc_ops / py_ops
        speedups[name] = sp
        print(f"{name:<26}{py_ops:>15,.0f}{fc_ops:>16,.0f}{sp:>9.1f}x")

    # ---- end-to-end estimate -------------------------------------------
    # Self-play profile (given): net work 68%, python-chess ~25-30% of total.
    # ~19% of total is is_game_over(claim_draw=True); the rest of the
    # python-chess share is legal-gen + board copies/push.
    print("\nEnd-to-end self-play estimate (Amdahl):")
    term_sp = speedups["terminal_result"]
    move_sp = speedups["make_copy"]
    legal_sp = speedups["legal_gen"]
    # Apportion the python-chess share of total runtime.
    frac_term = 0.19          # is_game_over(claim_draw=True)
    frac_move = 0.06          # board copies + push (apply)
    frac_legal = 0.03         # legal move generation
    pc_total = frac_term + frac_move + frac_legal
    # Rust still leaves the python-side encode_move + glue; assume the replaced
    # ops keep ~15% python glue overhead (pyo3 call + list building).
    glue = 0.15
    new_term = frac_term * (glue + (1 - glue) / term_sp)
    new_move = frac_move * (glue + (1 - glue) / move_sp)
    new_legal = frac_legal * (glue + (1 - glue) / legal_sp)
    rest = 1.0 - pc_total
    new_total = rest + new_term + new_move + new_legal
    print(f"  python-chess share modelled : {pc_total*100:.0f}% "
          f"(term {frac_term*100:.0f}%, move {frac_move*100:.0f}%, legal {frac_legal*100:.0f}%)")
    print(f"  assumed residual py glue     : {glue*100:.0f}% of each replaced op")
    print(f"  estimated end-to-end speedup : {1/new_total:.2f}x "
          f"({(1-new_total)*100:.0f}% wall-time reduction)")


if __name__ == "__main__":
    main()
