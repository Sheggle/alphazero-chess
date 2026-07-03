"""Strong classic-AI bot for 3D Connect 4 ("Score Four", 4x4x4).

Pure game AI: negamax alpha-beta + a hand-tuned, line-based heuristic. NO net,
NO learning. Separate track from the AlphaZero code.

Heuristic (the user's proposal)
-------------------------------
Scan all 76 winning lines. For a non-terminal position, from player +1's frame:
  * a line with ONLY +1 beads and empties, k of them (k=1..3):  += W[k]
  * a line with ONLY -1 beads and empties, k of them:           -= W[k]
  * a MIXED line (both colours) or an empty line:               0  (dead/neutral)
Exactly 3 weights W[1..3]. They are scale-invariant (only ratios matter), so the
default bakes them normalised with W[3] fixed. Terminals: win = +/-MATE (scaled
by depth so faster wins / slower losses are preferred); draw = 0.

Engine
------
Everything runs on a compact bitboard: two 64-bit ints (one bitmask per player,
bit `cell` set iff that player occupies it), a 16-int column-height array, and a
per-line pair of bead counts (cnt1/cnt2) so the heuristic is maintained
INCREMENTALLY on make/undo (touch only the ~7 lines through the placed cell) and
read in O(1) at a leaf. Cell index == env's flat `_cell` == `col*4 + z`, so a
returned column is directly an env action.

  * negamax alpha-beta, side-to-move frame, exact terminal detection;
  * move ordering: TT move, then children ranked by the resulting heuristic
    (immediate wins are detected and short-circuited before the loop);
  * iterative deepening with a wall-clock budget -> returns the best move of the
    last fully completed depth;
  * a transposition table keyed on the two bitboards (mate scores stored
    node-relative so they stay correct across depths).

Public API
----------
  Connect4AB(weights=DEFAULT_WEIGHTS).best_move(game, time_budget=1.0) -> (col, info)
  Connect4AB(...).best_move_depth(game, depth)                         -> (col, info)
  best_move(game, time_budget=1.0)          # module-level convenience
  best_move_depth(game, depth)
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from .connect4_env import _LINE_IDX  # 76 tuples of flat cell indices (0..63)

# --- precomputed line geometry (line index <-> cells; lines through each cell) ---
LINE_CELLS: list[tuple[int, ...]] = list(_LINE_IDX)
NUM_LINES = len(LINE_CELLS)  # 76
CELL_LINES: list[tuple[int, ...]] = [() for _ in range(64)]
_tmp: list[list[int]] = [[] for _ in range(64)]
for _li, _cells in enumerate(LINE_CELLS):
    for _c in _cells:
        _tmp[_c].append(_li)
CELL_LINES = [tuple(v) for v in _tmp]
del _tmp

# --- search constants ---
MATE = 1_000_000        # magnitude of a terminal win/loss
MATE_TH = 900_000       # scores beyond this are treated as mate distances
INF = 1 << 30
_EXACT, _LOWER, _UPPER = 0, 1, 2
_TT_CAP = 1_500_000

# Default tuned weights (W1, W2, W3). Baked from the depth-4 round-robin sweep
# (see tune_connect4_ab.py). W3 is normalised to 1.0; only ratios matter. The
# strength surface is a broad plateau around these values, but the GRADED
# structure is essential: flat (1,1,1) or degenerate (~0,1,1) weights score only
# ~6% against this setting.
DEFAULT_WEIGHTS = (0.096, 0.3825, 1.0)


class _Timeout(Exception):
    pass


class Connect4AB:
    def __init__(self, weights=DEFAULT_WEIGHTS, use_tt: bool = True):
        w1, w2, w3 = weights
        # Wp[k] = value of a pure line with k of our beads. k=0 and k=4 unused at
        # leaves (empty contributes 0; a 4-line is a terminal handled as mate).
        self.Wp = (0.0, float(w1), float(w2), float(w3), 0.0)
        self.weights = (float(w1), float(w2), float(w3))
        self.use_tt = use_tt
        self.tt: dict[tuple[int, int], tuple[int, int, float, int]] = {}
        # mutable search state (set by _init_from_game)
        self.bb0 = 0
        self.bb1 = 0
        self.h = [0] * 16
        self.cnt1 = [0] * NUM_LINES
        self.cnt2 = [0] * NUM_LINES
        self.score = 0.0
        self.turn = 0            # 0 -> player +1 to move, 1 -> player -1
        self.nodes = 0
        self.deadline = float("inf")
        self._root_best = -1

    # ------------------------------------------------------------------ setup
    def _init_from_game(self, game) -> None:
        flat = np.asarray(game.board).reshape(-1)  # C-order: idx == x*16+y*4+z == cell
        bb0 = bb1 = 0
        h = [0] * 16
        for cell in range(64):
            v = int(flat[cell])
            if v == 1:
                bb0 |= 1 << cell
            elif v == -1:
                bb1 |= 1 << cell
        for col in range(16):
            base = col * 4
            filled = 0
            for z in range(4):
                if flat[base + z] != 0:
                    filled = z + 1
            h[col] = filled
        cnt1 = [0] * NUM_LINES
        cnt2 = [0] * NUM_LINES
        score = 0.0
        Wp = self.Wp
        for li, cells in enumerate(LINE_CELLS):
            a = b = 0
            for c in cells:
                v = int(flat[c])
                if v == 1:
                    a += 1
                elif v == -1:
                    b += 1
            cnt1[li] = a
            cnt2[li] = b
            if a and not b:
                score += Wp[a]
            elif b and not a:
                score -= Wp[b]
        self.bb0, self.bb1, self.h = bb0, bb1, h
        self.cnt1, self.cnt2, self.score = cnt1, cnt2, score
        self.turn = 0 if game.to_play == 1 else 1

    # ------------------------------------------------------------ make / undo
    def _make(self, col: int) -> None:
        """Play `col` for the side to move; update bitboards + incremental score.

        Only called for NON-winning moves (immediate wins are short-circuited),
        so no terminal bookkeeping is needed here.
        """
        z = self.h[col]
        cell = col * 4 + z
        Wp = self.Wp
        cnt1 = self.cnt1
        cnt2 = self.cnt2
        if self.turn == 0:
            self.bb0 |= 1 << cell
            s = self.score
            for li in CELL_LINES[cell]:
                b = cnt2[li]
                if b == 0:                     # pure-ours line: W[a] -> W[a+1]
                    a = cnt1[li]
                    s += Wp[a + 1] - Wp[a]
                    cnt1[li] = a + 1
                else:                          # opp present
                    a = cnt1[li]
                    if a == 0:                 # was pure-opp -> now mixed: drop -W[b]
                        s += Wp[b]
                    cnt1[li] = a + 1
            self.score = s
        else:
            self.bb1 |= 1 << cell
            s = self.score
            for li in CELL_LINES[cell]:
                a = cnt1[li]
                if a == 0:                     # pure-ours(opp) line: -W[b] -> -W[b+1]
                    b = cnt2[li]
                    s -= Wp[b + 1] - Wp[b]
                    cnt2[li] = b + 1
                else:
                    b = cnt2[li]
                    if b == 0:                 # was pure-+1 -> now mixed: drop +W[a]
                        s -= Wp[a]
                    cnt2[li] = b + 1
            self.score = s
        self.h[col] = z + 1
        self.turn ^= 1

    def _undo(self, col: int) -> None:
        self.turn ^= 1
        z = self.h[col] - 1
        self.h[col] = z
        cell = col * 4 + z
        Wp = self.Wp
        cnt1 = self.cnt1
        cnt2 = self.cnt2
        if self.turn == 0:
            self.bb0 &= ~(1 << cell)
            s = self.score
            for li in CELL_LINES[cell]:
                b = cnt2[li]
                a = cnt1[li] - 1
                cnt1[li] = a
                if b == 0:
                    s -= Wp[a + 1] - Wp[a]
                elif a == 0:                   # back to pure-opp: restore -W[b]
                    s -= Wp[b]
            self.score = s
        else:
            self.bb1 &= ~(1 << cell)
            s = self.score
            for li in CELL_LINES[cell]:
                a = cnt1[li]
                b = cnt2[li] - 1
                cnt2[li] = b
                if a == 0:
                    s += Wp[b + 1] - Wp[b]
                elif b == 0:                   # back to pure-+1: restore +W[a]
                    s += Wp[a]
            self.score = s

    # ----------------------------------------------------------------- search
    def _order(self, moves: list[int], ttmove: int) -> list[int]:
        sign = 1.0 if self.turn == 0 else -1.0
        scored = []
        for col in moves:
            if col == ttmove:
                scored.append((1e18, col))
                continue
            self._make(col)
            scored.append((self.score * sign, col))  # mover-frame heuristic of child
            self._undo(col)
        scored.sort(reverse=True)
        return [c for _, c in scored]

    def _nm(self, depth: int, alpha: float, beta: float, ply: int) -> float:
        self.nodes += 1
        if (self.nodes & 8191) == 0 and perf_counter() > self.deadline:
            raise _Timeout

        key = (self.bb0, self.bb1)
        ttmove = -1
        if self.use_tt:
            e = self.tt.get(key)
            if e is not None:
                td, tflag, tval, tmove = e
                ttmove = tmove
                if td >= depth:
                    v = tval
                    if v > MATE_TH:
                        v -= ply
                    elif v < -MATE_TH:
                        v += ply
                    if tflag == _EXACT:
                        return v
                    if tflag == _LOWER:
                        if v > alpha:
                            alpha = v
                    elif tflag == _UPPER:
                        if v < beta:
                            beta = v
                    if alpha >= beta:
                        return v

        h = self.h
        moves = [c for c in range(16) if h[c] < 4]
        if not moves:
            return 0.0  # board full -> draw

        # immediate win for the side to move?
        cntp = self.cnt1 if self.turn == 0 else self.cnt2
        for col in moves:
            cell = col * 4 + h[col]
            for li in CELL_LINES[cell]:
                if cntp[li] == 3:
                    if ply == 0:
                        self._root_best = col
                    return MATE - ply

        if depth == 0:
            return self.score if self.turn == 0 else -self.score

        ordered = self._order(moves, ttmove)
        a0 = alpha
        best = -INF
        bestcol = ordered[0]
        for col in ordered:
            self._make(col)
            val = -self._nm(depth - 1, -beta, -alpha, ply + 1)
            self._undo(col)
            if val > best:
                best = val
                bestcol = col
                if val > alpha:
                    alpha = val
            if alpha >= beta:
                break
        if ply == 0:
            self._root_best = bestcol

        if self.use_tt:
            v = best
            if v > MATE_TH:
                v += ply
            elif v < -MATE_TH:
                v -= ply
            if best <= a0:
                flag = _UPPER
            elif best >= beta:
                flag = _LOWER
            else:
                flag = _EXACT
            if len(self.tt) >= _TT_CAP:
                self.tt.clear()
            self.tt[key] = (depth, flag, v, bestcol)
        return best

    # ------------------------------------------------------------------- API
    def best_move_depth(self, game, depth: int):
        """Fixed-depth search (no time limit). Returns (col, info)."""
        self._init_from_game(game)
        self.deadline = float("inf")
        self.nodes = 0
        self._root_best = -1
        legal = game.legal_moves()
        if not legal:
            return -1, {"depth": 0, "nodes": 0, "score": 0.0, "reason": "no legal moves"}
        if len(legal) == 1:
            return legal[0], {"depth": 0, "nodes": 0, "score": 0.0, "reason": "forced"}
        t0 = perf_counter()
        score = self._nm(depth, -INF, INF, 0)
        col = self._root_best if self._root_best >= 0 else self._fallback(game)
        return col, {"depth": depth, "nodes": self.nodes, "score": score,
                     "time": perf_counter() - t0}

    def best_move(self, game, time_budget: float = 1.0):
        """Iterative-deepening search within `time_budget` seconds.

        Returns (col, info). Always returns a legal move even for a tiny budget.
        """
        self._init_from_game(game)
        legal = game.legal_moves()
        if not legal:
            return -1, {"depth": 0, "nodes": 0, "score": 0.0, "reason": "no legal moves"}
        if len(legal) == 1:
            return legal[0], {"depth": 0, "nodes": 0, "score": 0.0, "reason": "forced",
                              "time": 0.0}

        t0 = perf_counter()
        self.deadline = t0 + time_budget
        self.nodes = 0
        best = self._fallback(game)          # guaranteed-legal fallback
        best_score = 0.0
        depth_reached = 0
        empties = sum(1 for c in range(16) if self.h[c] < 4) * 1  # >= max useful depth bound
        empties = sum(4 - self.h[c] for c in range(16))
        max_depth = min(empties, 64)

        for depth in range(1, max_depth + 1):
            self._root_best = -1
            try:
                score = self._nm(depth, -INF, INF, 0)
            except _Timeout:
                break
            if self._root_best >= 0:
                best = self._root_best
                best_score = score
                depth_reached = depth
            if abs(score) > MATE_TH:          # solved (forced win/loss found)
                break

        return best, {"depth": depth_reached, "nodes": self.nodes,
                      "score": best_score, "time": perf_counter() - t0}

    def _fallback(self, game) -> int:
        """Cheap always-legal move: take a win, else block, else centre-most."""
        me = game.to_play
        wins = game.winning_columns(me)
        if wins:
            return wins[0]
        blocks = game.winning_columns(-me)
        if blocks:
            return blocks[0]
        legal = game.legal_moves()
        # centre columns (x,y in {1,2}) pass through the most lines
        center = [1 * 4 + 1, 1 * 4 + 2, 2 * 4 + 1, 2 * 4 + 2]
        for c in center:
            if c in legal:
                return c
        return legal[0]


# ------------------------------------------------------------- module-level API
_SHARED = Connect4AB()


def best_move(game, time_budget: float = 1.0):
    return _SHARED.best_move(game, time_budget=time_budget)


def best_move_depth(game, depth: int):
    return _SHARED.best_move_depth(game, depth)
