"""Always-on, fast evaluation harness for AlphaZero-chess checkpoints.

Goal: for any checkpoint, tell us (1) how strong it is and (2) *where it fails
in ways we wouldn't expect*. Every metric is a plain function taking a
`ChessEvaluator` (+ knobs) and returning numbers, so the harness composes and
can run on every training checkpoint in well under a minute.

Metrics
-------
- `tactics`            : frozen mate-in-1 / hanging-capture suite solve rates.
- `tactics_scan`       : same rates + per-position detail (used for failures).
- `play_vs_random`     : score vs a random mover *and* the net's own hang rate,
                         measured in one shared game loop (cheap).
- `value_calibration`  : does the value head track material / rollout outcome?
- `unexpected_failures`: concrete FENs where the net throws away material it did
                         not need to, or misses a mate-in-1 it had.

Conventions reused from the project: action indices are the AlphaZero 4672
scheme; `ChessGame` is canonical (White-up); the value head is from the
side-to-move's perspective. Material is scored White-positive then signed by the
side to move where a side-relative number is wanted.
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import chess
import numpy as np

from .chess_env import ChessGame, encode_move
from .gumbel import GumbelMCTS
from .chess_tactics import load_suite, tactics_rates

# Piece values (King = 0; it can't "hang" in the material sense).
_VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


# --------------------------------------------------------------------------- #
# small shared helpers
# --------------------------------------------------------------------------- #

def _action_to_uci(board: chess.Board, action: int) -> str | None:
    for mv in board.legal_moves:
        if encode_move(board, mv) == action:
            return mv.uci()
    return None


def _greedy_action(evaluator, state, sims: int, mc: int, rng) -> int:
    """Deterministic best move (no Gumbel noise), as used in eval/self-play."""
    a, _ = GumbelMCTS(evaluator, n_sims=sims, max_considered=mc, rng=rng).run(
        state, add_noise=False)
    return int(a)


def material_white(board: chess.Board) -> int:
    """Material balance in points, White-positive."""
    d = 0
    for _, p in board.piece_map().items():
        v = _VAL.get(p.piece_type, 0)
        d += v if p.color == chess.WHITE else -v
    return d


def _adjudicate_white(board: chess.Board, mat_thresh: float) -> int:
    """Outcome from White's perspective: terminal result, else material-capped."""
    if board.is_game_over(claim_draw=True):
        oc = board.outcome(claim_draw=True)
        if oc is None or oc.winner is None:
            return 0
        return 1 if oc.winner == chess.WHITE else -1
    d = material_white(board)
    return 1 if d >= mat_thresh else (-1 if d <= -mat_thresh else 0)


def hanging_pieces(board: chess.Board, color: bool, min_value: int = 3) -> list[int]:
    """Squares of `color`'s pieces (value >= min_value) that are *hanging*:
    attacked by the opponent and not defended by a friendly piece — i.e. a free
    capture is available to the opponent on the move.

    Uses python-chess `is_attacked_by`, which counts pseudo-legal attackers
    (a pinned attacker/defender still counts). That makes this a fast, slightly
    conservative proxy rather than a full static-exchange evaluation.
    """
    opp = not color
    out = []
    for sq, pc in board.piece_map().items():
        if pc.color != color or _VAL[pc.piece_type] < min_value:
            continue
        if board.is_attacked_by(opp, sq) and not board.is_attacked_by(color, sq):
            out.append(sq)
    return out


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    sx, sy = x.std(), y.std()
    if sx == 0 or sy == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


# --------------------------------------------------------------------------- #
# 1. tactics
# --------------------------------------------------------------------------- #

def tactics(evaluator, sims: int = 32, max_considered: int = 8) -> dict:
    """Frozen-suite solve rates (wraps `tactics_rates`)."""
    return tactics_rates(evaluator, sims=sims, max_considered=max_considered)


def tactics_scan(evaluator, sims: int = 32, max_considered: int = 8,
                 rng_seed: int = 0) -> dict:
    """Like `tactics` but also returns per-position detail (chosen move vs the
    accepted solutions). Lets us compute solve rates *and* harvest concrete
    failure examples from a single pass over the suite."""
    suite = load_suite()
    solved = {"mate_in_1": 0, "hanging_capture": 0}
    total = {"mate_in_1": 0, "hanging_capture": 0}
    detail = []
    for entry in suite:
        typ = entry["type"]
        total[typ] = total.get(typ, 0) + 1
        board = chess.Board(entry["fen"])
        a = _greedy_action(evaluator, ChessGame(board), sims, max_considered,
                           np.random.default_rng(rng_seed))
        uci = _action_to_uci(board, a)
        ok = uci is not None and uci in set(entry["solutions"])
        if ok:
            solved[typ] = solved.get(typ, 0) + 1
        detail.append({"fen": entry["fen"], "type": typ, "net_move": uci,
                       "solutions": entry["solutions"], "solved": ok})

    def rate(s, t):
        return s / t if t else 0.0

    n_overall = sum(total.values())
    s_overall = sum(solved.values())
    rates = {
        "mate_in_1": rate(solved.get("mate_in_1", 0), total.get("mate_in_1", 0)),
        "hanging_capture": rate(solved.get("hanging_capture", 0),
                                total.get("hanging_capture", 0)),
        "overall": rate(s_overall, n_overall),
        "n": {"mate_in_1": total.get("mate_in_1", 0),
              "hanging_capture": total.get("hanging_capture", 0),
              "overall": n_overall},
    }
    return {"rates": rates, "detail": detail}


# --------------------------------------------------------------------------- #
# 2 + 4. vs-random score AND own hang/blunder rate, in one game loop
# --------------------------------------------------------------------------- #

def play_vs_random(evaluator, games: int = 40, sims: int = 16,
                   max_considered: int = 8, max_ply: int = 100,
                   mat_thresh: float = 1.0, min_value: int = 3,
                   seed: int = 12345) -> dict:
    """Play `games` vs a random mover (alternating colors, capped + material-
    adjudicated) and, in the same loop, measure how often the net's *own* moves
    leave one of its >= `min_value` pieces hanging (a free capture for the
    opponent next move). The hang rate is a core "does it give away material"
    signal that — unlike the vs-random score — does not saturate.

    Returns vs_random stats, avg game length, and:
      hang_move_rate : fraction of the net's moves after which it has a hanging
                       piece (post-move state, opponent to move).
      blunder_rate   : fraction of the net's moves that *newly* create a hang
                       (the piece was not already hanging before the move).
    """
    rand = random.Random(seed)
    wins = draws = losses = 0
    lengths = []
    net_moves = 0
    hang_moves = 0
    new_hang_moves = 0

    for i in range(games):
        net_is_white = (i % 2 == 0)
        g = ChessGame()
        while not g.is_terminal() and g.ply < max_ply:
            net_to_move = (g.to_play == 1) == net_is_white
            if net_to_move:
                net_color = chess.WHITE if net_is_white else chess.BLACK
                pre = set(hanging_pieces(g.board, net_color, min_value))
                a = _greedy_action(evaluator, g, sims, max_considered,
                                   np.random.default_rng(seed + i))
                g = g.apply(a)
                post = set(hanging_pieces(g.board, net_color, min_value))
                net_moves += 1
                if post:
                    hang_moves += 1
                if post - pre:  # a piece that was safe is now hanging
                    new_hang_moves += 1
            else:
                legal = g.legal_moves()
                g = g.apply(legal[rand.randrange(len(legal))])
        zc = _adjudicate_white(g.board, mat_thresh)
        net_z = zc if net_is_white else -zc
        wins += net_z > 0
        draws += net_z == 0
        losses += net_z < 0
        lengths.append(g.ply)

    n = max(games, 1)
    return {
        "score": (wins + 0.5 * draws) / n,
        "wins": wins, "draws": draws, "losses": losses,
        "avg_len": sum(lengths) / n,
        "net_moves": net_moves,
        "hang_move_rate": hang_moves / net_moves if net_moves else 0.0,
        "blunder_rate": new_hang_moves / net_moves if net_moves else 0.0,
    }


# --------------------------------------------------------------------------- #
# 3. value-head calibration
# --------------------------------------------------------------------------- #

def _random_positions(n_positions: int, max_ply: int, seed: int,
                      min_ply: int = 6) -> list[chess.Board]:
    """Sample non-terminal positions from random games (decorrelated from the
    net). Skips the first `min_ply` plies so positions aren't all openings."""
    rnd = random.Random(seed)
    out = []
    while len(out) < n_positions:
        board = chess.Board()
        ply = 0
        while not board.is_game_over(claim_draw=True) and ply < max_ply:
            if ply >= min_ply and len(out) < n_positions and rnd.random() < 0.25:
                out.append(board.copy(stack=False))
            moves = list(board.legal_moves)
            board.push(moves[rnd.randrange(len(moves))])
            ply += 1
    return out[:n_positions]


def _random_rollout_result(board: chess.Board, rollout_len: int,
                           mat_thresh: float, rnd: random.Random) -> int:
    """Adjudicated outcome (White perspective) after a short random rollout."""
    b = board.copy(stack=False)
    steps = 0
    while not b.is_game_over(claim_draw=True) and steps < rollout_len:
        moves = list(b.legal_moves)
        b.push(moves[rnd.randrange(len(moves))])
        steps += 1
    return _adjudicate_white(b, mat_thresh)


def value_calibration(evaluator, n_positions: int = 120, max_ply: int = 80,
                      rollout_len: int = 20, mat_thresh: float = 1.0,
                      mat_scale: float = 5.0, seed: int = 7) -> dict:
    """Correlate the value head with (a) current material and (b) the eventual
    material-adjudicated result of a short random rollout — both from the
    side-to-move's perspective (matching the value head's frame).

    Pearson r is scale-free. MAE is reported against comparable targets:
    `mae_vs_material` against tanh(material/mat_scale) (a [-1,1] proxy), and
    `mae_vs_rollout` against the rollout result in {-1,0,1}. A net that has
    learned material shows positive r on both.
    """
    boards = _random_positions(n_positions, max_ply, seed)
    rnd = random.Random(seed + 1)
    vals, mats, rolls = [], [], []
    for board in boards:
        stm_sign = 1 if board.turn == chess.WHITE else -1
        _, value = evaluator.predict(ChessGame(board.copy(stack=False)))
        mat_stm = material_white(board) * stm_sign
        roll_stm = _random_rollout_result(board, rollout_len, mat_thresh, rnd) * stm_sign
        vals.append(value)
        mats.append(mat_stm)
        rolls.append(roll_stm)

    v = np.array(vals, dtype=np.float64)
    m = np.array(mats, dtype=np.float64)
    r = np.array(rolls, dtype=np.float64)
    mat_target = np.tanh(m / mat_scale)
    return {
        "n": len(boards),
        "r_material": _pearson(v, m),
        "r_rollout": _pearson(v, r),
        "mae_vs_material": float(np.mean(np.abs(v - mat_target))),
        "mae_vs_rollout": float(np.mean(np.abs(v - r))),
        "value_mean": float(v.mean()),
        "value_std": float(v.std()),
    }


# --------------------------------------------------------------------------- #
# 5. unexpected-failure finder
# --------------------------------------------------------------------------- #

def unexpected_failures(evaluator, scan: dict | None = None,
                        sims: int = 32, max_considered: int = 8,
                        n_hang_positions: int = 60, hang_sims: int = 16,
                        max_ply: int = 80, min_value: int = 3,
                        max_examples: int = 4, seed: int = 11) -> dict:
    """Surface concrete positions where the net fails surprisingly:

    - missed_mate_in_1 : suite positions with a mate-in-1 available that the net
                         did not play (it *had* a forced win and missed it).
    - missed_hanging_capture : suite free-capture wins the net declined.
    - self_hangs       : sampled positions where the net's greedy move hangs a
                         piece (>= min_value) *and* a legal move existed that did
                         not create that hang — i.e. avoidable material loss.

    Reuses a precomputed `tactics_scan` result if given (avoids re-searching).
    """
    if scan is None:
        scan = tactics_scan(evaluator, sims=sims, max_considered=max_considered)
    missed_mate = []
    missed_hang = []
    for d in scan["detail"]:
        if d["solved"]:
            continue
        rec = {"fen": d["fen"], "net_move": d["net_move"],
               "best_moves": d["solutions"]}
        if d["type"] == "mate_in_1" and len(missed_mate) < max_examples:
            missed_mate.append(rec)
        elif d["type"] == "hanging_capture" and len(missed_hang) < max_examples:
            missed_hang.append(rec)

    # Sampled avoidable self-hangs.
    self_hangs = []
    boards = _random_positions(n_hang_positions, max_ply, seed)
    for board in boards:
        if len(self_hangs) >= max_examples:
            break
        color = board.turn
        g = ChessGame(board.copy(stack=False))
        a = _greedy_action(evaluator, g, hang_sims, max_considered,
                           np.random.default_rng(seed))
        uci = _action_to_uci(board, a)
        g2 = g.apply(a)
        hung = hanging_pieces(g2.board, color, min_value)
        if not hung:
            continue
        # Did a safe alternative exist? Find one legal move that creates no new
        # hang relative to the pre-move position.
        pre = set(hanging_pieces(board, color, min_value))
        safe = None
        for mv in board.legal_moves:
            if mv.uci() == uci:
                continue
            b2 = board.copy(stack=False)
            b2.push(mv)
            if not (set(hanging_pieces(b2, color, min_value)) - pre):
                safe = mv.uci()
                break
        if safe is not None:
            # `hung` squares are on the post-move board (g2), so read the piece
            # values there — the net's move may itself have moved a piece onto a
            # hanging square, in which case the pre-move board has nothing there.
            lost = max(_VAL[g2.board.piece_type_at(sq)] for sq in hung)
            self_hangs.append({"fen": board.fen(), "net_move": uci,
                               "hangs_value": lost, "safe_alt": safe})

    return {"missed_mate_in_1": missed_mate,
            "missed_hanging_capture": missed_hang,
            "self_hangs": self_hangs}


# --------------------------------------------------------------------------- #
# top-level driver
# --------------------------------------------------------------------------- #

def evaluate(evaluator, *, tactics_sims: int = 32, vs_random_games: int = 24,
             vs_random_sims: int = 10, vs_random_max_ply: int = 50,
             calib_positions: int = 100, max_considered: int = 8) -> dict:
    """Run the full harness and return a single metrics dict (JSON-ready).

    Defaults are tuned to finish in well under a minute on one CPU thread:
    tactics uses the full self-play sim budget (32, the precise headline metric);
    the vs-random loop is the bulk of the cost so it runs at a reduced sim budget
    and a 60-ply cap. The hang/blunder rate and tactics are the high-signal,
    low-variance numbers; vs-random score here carries ~0.09 stderr.
    """
    scan = tactics_scan(evaluator, sims=tactics_sims,
                        max_considered=max_considered)
    return {
        "tactics": scan["rates"],
        "vs_random": play_vs_random(evaluator, games=vs_random_games,
                                    sims=vs_random_sims, max_ply=vs_random_max_ply,
                                    max_considered=max_considered),
        "value_calibration": value_calibration(evaluator,
                                                n_positions=calib_positions),
        "failures": unexpected_failures(evaluator, scan=scan, hang_sims=10,
                                        n_hang_positions=40,
                                        max_considered=max_considered),
    }
