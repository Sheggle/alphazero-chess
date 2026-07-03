"""Batched Gumbel self-play: run B games concurrently, evaluate every pending
search leaf in ONE network forward.

Why this exists
---------------
Profiling shows the network forward is ~68% of self-play wall time and it is run
as *batch-1* (one position per call). The single biggest throughput win is to
batch leaf evaluations across many independent games: advance each active game's
search by one simulation, collect the (game -> leaf) that needs a network value,
do a *single* `net()` over all of them, scatter priors/values back, repeat.

To keep the training data identical to the existing single-game pipeline this
module reuses the *exact* Gumbel acting + Sequential-Halving schedule and the
*exact* completed-Q policy target from `alphazero.gumbel.GumbelMCTS`. The only
structural changes are mechanical and value-preserving:

  * **Coroutine restructure.** Each game's search is a generator that *yields* a
    leaf needing evaluation and is *resumed* with (priors, value). A scheduler
    drives B of them and batches whatever they yield. The math is unchanged.

  * **Lazy child expansion.** `GumbelMCTS._expand` eagerly builds a child board
    for every legal move (~35), almost all never visited. Here a node stores the
    prior vector and materialises a child node only when PUCT first selects it.
    An unvisited child has n=0, q=0 and a known prior -- identical to the eager
    node before its first visit -- so selection is bit-for-bit the same.

  * **Make / unmake.** A single working `chess.Board` is pushed down the descent
    and popped on the way back up, instead of copying a board per node. One board
    copy per *move* (the search root) replaces ~35-per-expansion copies.

  * **Cheap in-search terminal test.** `is_game_over(claim_draw=True)` (python-
    chess threefold scan, ~19% of time) is replaced *inside the search* by the
    cheap `claim_draw=False` automatic-terminal test. The full
    `outcome(claim_draw=True)` + material adjudication is still used at *game
    end* to label samples, so training targets are unaffected. (Set
    `in_search_claim_draw=True` to recover exact `GumbelMCTS` search semantics --
    the test suite uses this to prove equivalence.)

Correctness: with `in_search_claim_draw=True` and a 1-leaf batch, this engine
reproduces `GumbelMCTS.run` *exactly* (same action and same policy vector) given
the same net and seed -- see `tests/test_chess_batched.py`. The production path
(batch>1) differs only by the network's own batch-vs-single floating-point noise.

Output is the same sparse `Sample` tuple as `chess_train.play_chess_game`:
`(planes float16 (18,8,8), pi_indices int16, pi_values float32, z float32)`,
plus a per-game stats dict.

One semantic improvement over the legacy engine: because the working board keeps
full move history, threefold repetition is actually detectable at game end
(legacy `ChessGame.apply` truncates the move stack to one ply, so it never was).
This makes a few shuffling games draw instead of running to the ply cap -- it is
more correct, and it never affects the equivalence tests (those positions do not
reach a threefold within a shallow search).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import chess
import numpy as np
import torch

from .chess_encode import encode_board
from .chess_env import ACTION_SIZE, encode_move

# Material values for end-of-game adjudication -- identical to chess_train.
_VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def material_diff(board: chess.Board) -> int:
    """Material balance from White's perspective (matches chess_train)."""
    d = 0
    for _, p in board.piece_map().items():
        v = _VAL.get(p.piece_type, 0)
        d += v if p.color == chess.WHITE else -v
    return d


def _result_white(board: chess.Board) -> int:
    """Terminal outcome from White's perspective via outcome(claim_draw=True)."""
    oc = board.outcome(claim_draw=True)
    if oc is None or oc.winner is None:
        return 0
    return 1 if oc.winner == chess.WHITE else -1


def _outcome_white(board: chess.Board, mat_thresh: float) -> int:
    """White-perspective game label: real result if terminal, else material
    adjudication (identical to chess_train._outcome_white)."""
    if board.is_game_over(claim_draw=True):
        return _result_white(board)
    d = material_diff(board)
    return 1 if d >= mat_thresh else (-1 if d <= -mat_thresh else 0)


class _BoardView:
    """Minimal object exposing `.board` so `encode_board` can read a raw board
    without us constructing/copying a full ChessGame."""

    __slots__ = ("board",)

    def __init__(self, board: chess.Board):
        self.board = board


class _Node:
    """Lazy MCTS node. Children are materialised on first selection; the board is
    never stored (reconstructed via make/unmake on the shared working board)."""

    __slots__ = (
        "prior", "to_play", "n", "w", "children", "is_expanded",
        "is_terminal", "term_value", "legal_actions", "legal_moves", "priors",
        "action_pos",
    )

    def __init__(self, prior: float, to_play: int):
        self.prior = prior
        self.to_play = to_play
        self.n = 0
        self.w = 0.0
        self.children: dict[int, "_Node"] = {}
        self.is_expanded = False
        self.is_terminal = None          # None = undetermined, else bool
        self.term_value = 0.0            # in this node's to-move perspective
        self.legal_actions: list[int] | None = None
        self.legal_moves: list[chess.Move] | None = None
        self.priors: np.ndarray | None = None   # aligned to legal_actions
        self.action_pos: dict[int, int] | None = None  # root only

    @property
    def q(self) -> float:
        return self.w / self.n if self.n > 0 else 0.0


# --------------------------------------------------------------------------
# Eval request: what a paused search hands to the scheduler.
# --------------------------------------------------------------------------
@dataclass
class _EvalReq:
    planes: np.ndarray          # (18,8,8) float32
    legal: list                 # legal action indices, defines prior alignment


class _BatchedGumbel:
    """Engine holding the (stateless) Gumbel hyper-parameters. One instance is
    shared by all concurrent games; per-game state lives in the generators."""

    def __init__(self, n_sims=32, max_considered=8, c_visit=50.0, c_scale=1.0,
                 c_puct=1.5, in_search_claim_draw=False):
        self.n_sims = n_sims
        self.max_considered = max_considered
        self.c_visit = c_visit
        self.c_scale = c_scale
        self.c_puct = c_puct
        self.in_search_claim_draw = in_search_claim_draw

    # ---- terminal helpers (cheap in-search; full version used at game end) ----

    def _is_terminal(self, board: chess.Board) -> bool:
        return board.is_game_over(claim_draw=self.in_search_claim_draw)

    def _terminal_value(self, board: chess.Board, to_play: int) -> float:
        oc = board.outcome(claim_draw=self.in_search_claim_draw)
        if oc is None or oc.winner is None:
            white = 0
        else:
            white = 1 if oc.winner == chess.WHITE else -1
        return float(white * to_play)

    # ---- expansion (a generator: yields a leaf, resumes with priors+value) ----

    def _expand(self, node: _Node, wb: chess.Board):
        moves = list(wb.legal_moves)
        actions = [encode_move(wb, mv) for mv in moves]
        planes = encode_board(_BoardView(wb))
        priors, value = yield _EvalReq(planes, actions)   # <-- batched here
        node.legal_moves = moves
        node.legal_actions = actions
        node.priors = np.asarray(priors, dtype=np.float64)  # aligned to actions
        node.children = {}
        node.is_expanded = True
        node.is_terminal = False
        return float(value)

    # ---- PUCT child selection (mirrors AZMCTS._select_child, lazily) ----

    def _select_child(self, node: _Node):
        sqrt_n = math.sqrt(node.n)
        best_j, best_a, best_score = -1, None, -math.inf
        la = node.legal_actions
        pri = node.priors
        ch = node.children
        cp = self.c_puct
        for j in range(len(la)):
            a = la[j]
            child = ch.get(a)
            if child is None:
                cn, cq = 0, 0.0
            else:
                cn, cq = child.n, child.q
            u = cp * pri[j] * sqrt_n / (1 + cn)
            score = -cq + u
            if score > best_score:          # strict '>' -> ties keep legal order
                best_score, best_j, best_a = score, j, a
        child = ch.get(best_a)
        if child is None:
            child = _Node(prior=float(node.priors[best_j]), to_play=-node.to_play)
            ch[best_a] = child
        return child, node.legal_moves[best_j]

    # ---- one simulation forced through root->action (mirrors _simulate) ----

    def _simulate(self, root: _Node, wb: chess.Board, action: int):
        pushed = 0
        j = root.action_pos[action]
        mv = root.legal_moves[j]
        wb.push(mv); pushed += 1
        child = root.children.get(action)
        if child is None:
            child = _Node(prior=float(root.priors[j]), to_play=-root.to_play)
            root.children[action] = child

        path = [child]
        node = child
        while node.is_expanded and not node.is_terminal:
            nxt, mv2 = self._select_child(node)
            wb.push(mv2); pushed += 1
            path.append(nxt)
            node = nxt

        leaf = node
        if leaf.is_terminal is None:
            if self._is_terminal(wb):
                leaf.is_terminal = True
                leaf.term_value = self._terminal_value(wb, leaf.to_play)
            else:
                leaf.is_terminal = False

        if leaf.is_terminal:
            value = leaf.term_value
        else:                                   # non-terminal, unexpanded leaf
            value = yield from self._expand(leaf, wb)

        # negamax backup (leaf value is in leaf's to-move perspective)
        v = value
        for nd in reversed(path):
            nd.n += 1
            nd.w += v
            v = -v
        root.n += 1

        for _ in range(pushed):
            wb.pop()
        return value

    # ---- Sequential Halving over the considered root actions ----

    def _sequential_halving(self, root: _Node, wb: chess.Board, considered, gpref):
        considered = list(considered)
        budget = self.n_sims
        used = 0
        num_phases = (max(1, math.ceil(math.log2(len(considered))))
                      if len(considered) > 1 else 1)

        while used < budget and len(considered) >= 1:
            per = (max(1, (budget // num_phases) // len(considered))
                   if len(considered) > 1 else budget)
            for a in considered:
                for _ in range(per):
                    if used >= budget:
                        break
                    yield from self._simulate(root, wb, a)
                    used += 1
                if used >= budget:
                    break
            if len(considered) <= 1 or used >= budget:
                break
            considered.sort(
                key=lambda a: gpref[a] + self._sigma(root, self._q(root, a)),
                reverse=True)
            considered = considered[: max(1, len(considered) // 2)]

    # ---- value / policy helpers (identical formulas to GumbelMCTS) ----

    def _q(self, root: _Node, action: int) -> float:
        child = root.children.get(action)
        return -child.q if (child is not None and child.n > 0) else 0.0

    def _sigma(self, root: _Node, q: float) -> float:
        max_visit = max((c.n for c in root.children.values()), default=0)
        return (self.c_visit + max_visit) * self.c_scale * q

    def _completed_policy(self, root, root_value, logits, legal) -> np.ndarray:
        priors = np.exp(logits - logits.max())
        priors /= priors.sum()

        def cn(a):
            c = root.children.get(a)
            return c.n if c is not None else 0

        visited = [(i, a) for i, a in enumerate(legal) if cn(a) > 0]
        n_total = sum(cn(a) for a in legal)
        if visited:
            sum_p = sum(priors[i] for i, _ in visited)
            weighted_q = sum(priors[i] * self._q(root, a) for i, a in visited) / max(sum_p, 1e-12)
            v_mix = (root_value + n_total * weighted_q) / (1 + n_total)
        else:
            v_mix = root_value

        completed_q = np.empty(len(legal))
        for i, a in enumerate(legal):
            completed_q[i] = self._q(root, a) if cn(a) > 0 else v_mix

        score = logits + np.array([self._sigma(root, q) for q in completed_q])
        score -= score.max()
        ex = np.exp(score)
        probs = ex / ex.sum()

        pi = np.zeros(ACTION_SIZE, dtype=np.float32)
        for i, a in enumerate(legal):
            pi[a] = probs[i]
        return pi

    # ---- the full search for one position, as a resumable generator ----

    def search_gen(self, root_board: chess.Board, rng, add_noise: bool):
        """Generator: yields `_EvalReq`, resumes with (priors_aligned, value);
        returns (chosen_action, improved_policy). Mirrors GumbelMCTS.run."""
        wb = root_board   # owned, mutated via push/pop, restored each simulation
        root = _Node(prior=0.0,
                     to_play=(1 if wb.turn == chess.WHITE else -1))
        root_value = yield from self._expand(root, wb)
        root.action_pos = {a: i for i, a in enumerate(root.legal_actions)}
        root.n = 1
        root.w = root_value

        legal = root.legal_actions
        priors = np.clip(root.priors.astype(np.float64), 1e-12, 1.0)
        logits = np.log(priors)
        gumbel = (rng.gumbel(size=len(legal)) if add_noise
                  else np.zeros(len(legal)))

        m = min(self.max_considered, len(legal), max(2, self.n_sims))
        order = np.argsort(-(gumbel + logits))
        considered = [legal[i] for i in order[:m]]
        gpref = {legal[i]: gumbel[i] + logits[i] for i in range(len(legal))}

        yield from self._sequential_halving(root, wb, considered, gpref)

        improved = self._completed_policy(root, root_value, logits, legal)
        best = max(considered,
                   key=lambda a: gpref[a] + self._sigma(root, self._q(root, a)))
        return best, improved


class _Search:
    """A single game's in-flight search: drives one `search_gen` generator and
    exposes the leaf currently awaiting evaluation."""

    __slots__ = ("gen", "request", "done", "result")

    def __init__(self, engine: _BatchedGumbel, root_board, rng, add_noise):
        self.gen = engine.search_gen(root_board, rng, add_noise)
        self.request: _EvalReq | None = None
        self.done = False
        self.result = None
        # search_gen always yields the root leaf before doing anything, so
        # start() can never complete the generator.
        self.request = next(self.gen)

    def resume(self, priors, value):
        try:
            self.request = self.gen.send((priors, value))
        except StopIteration as e:
            self.done = True
            self.result = e.value
            self.request = None


# --------------------------------------------------------------------------
# Eval backend: one network forward over a list of requests.
# --------------------------------------------------------------------------
def _make_batch_eval(evaluator):
    """Build eval_fn(requests) -> (priors_list, values). priors_list[i] is the
    softmax over requests[i].legal logits (same math as ChessEvaluator.predict),
    aligned to that request's legal order."""
    net = evaluator.net
    device = evaluator.device

    @torch.no_grad()
    def eval_fn(requests):
        net.eval()
        x = torch.from_numpy(np.stack([r.planes for r in requests])).to(device)
        logits, values = net(x)
        logits = logits.cpu().numpy()
        values = values.cpu().numpy()
        priors_list = []
        for i, r in enumerate(requests):
            ll = logits[i][r.legal]
            ll = ll - ll.max()
            ex = np.exp(ll)
            priors_list.append(ex / ex.sum())
        return priors_list, values

    return eval_fn


def run_single_search(engine, board, rng, eval_fn, add_noise):
    """Drive ONE search to completion as 1-element batches. Used by the test
    suite: with `in_search_claim_draw=True` this matches GumbelMCTS.run exactly.
    Returns (action, improved_policy)."""
    s = _Search(engine, board.copy(), rng, add_noise)
    while not s.done:
        priors_list, values = eval_fn([s.request])
        s.resume(priors_list[0], float(values[0]))
    return s.result


# --------------------------------------------------------------------------
# Per-game bookkeeping for the concurrent driver.
# --------------------------------------------------------------------------
@dataclass
class _Game:
    board: chess.Board
    rng: object
    recs: list = field(default_factory=list)   # (planes, pi, to_play) per move
    search: _Search | None = None


def _action_to_move(board: chess.Board, action: int) -> chess.Move:
    for mv in board.legal_moves:
        if encode_move(board, mv) == action:
            return mv
    raise ValueError(f"action {action} not legal in this position")


def _finalize_game(g: _Game, mat_thresh: float):
    """Turn a completed game's records into Sample tuples + a stats dict."""
    board = g.board
    z_white = _outcome_white(board, mat_thresh)
    samples = []
    for planes, pi, to_play in g.recs:
        z = float(z_white * to_play)
        idx = np.nonzero(pi)[0].astype(np.int16)
        samples.append((planes.astype(np.float16), idx,
                        pi[idx].astype(np.float32), np.float32(z)))
    terminal = board.is_game_over(claim_draw=True)
    stats = {"terminal": terminal, "plies": board.ply(), "z_white": z_white,
             "result": _result_white(board) if terminal else None}
    return samples, stats


def play_batched_games(evaluator, n_games, concurrency, *, sims=32,
                       max_considered=8, c_visit=50.0, c_scale=1.0, c_puct=1.5,
                       max_ply=100, mat_thresh=1.0, seed=0,
                       in_search_claim_draw=False, add_noise=True):
    """Play `n_games` self-play games, keeping up to `concurrency` of them active
    at once and evaluating all their pending leaves in single batched forwards.

    Returns (samples, stats):
      * samples: flat list of `(planes f16 (18,8,8), pi_idx i16, pi_val f32,
        z f32)` -- identical layout to chess_train.play_chess_game.
      * stats:   one dict per game `{terminal, plies, z_white, result}`.

    Determinism: game i uses np.random.default_rng(SeedSequence(seed).spawn) so
    results are reproducible and independent of `concurrency` (the scheduling
    order does not touch any game's RNG)."""
    engine = _BatchedGumbel(sims, max_considered, c_visit, c_scale, c_puct,
                            in_search_claim_draw)
    eval_fn = _make_batch_eval(evaluator)

    seeds = list(np.random.SeedSequence(seed).spawn(n_games))
    next_idx = 0
    all_samples: list = []
    all_stats: list = []

    def spawn() -> _Game | None:
        nonlocal next_idx
        if next_idx >= n_games:
            return None
        g = _Game(board=chess.Board(),
                  rng=np.random.default_rng(seeds[next_idx]))
        next_idx += 1
        return g

    def playable(board: chess.Board) -> bool:
        return (not board.is_game_over(claim_draw=True)) and board.ply() < max_ply

    def begin_search(g: _Game):
        g.search = _Search(engine, g.board.copy(), g.rng, add_noise)

    active: list[_Game] = []
    # Initial fill.
    while len(active) < concurrency:
        g = spawn()
        if g is None:
            break
        if playable(g.board):
            begin_search(g)
            active.append(g)
        else:                                   # start pos is never terminal
            s, st = _finalize_game(g, mat_thresh)
            all_samples.extend(s); all_stats.append(st)

    while active:
        # Every active game has a leaf awaiting evaluation -> one batched forward.
        reqs = [g.search.request for g in active]
        priors_list, values = eval_fn(reqs)
        for g, pr, v in zip(active, priors_list, values):
            g.search.resume(pr, float(v))

        next_active: list[_Game] = []
        for g in active:
            if not g.search.done:
                next_active.append(g)
                continue
            # This game's move decision is complete: record + apply.
            action, pi = g.search.result
            planes = encode_board(_BoardView(g.board))
            to_play = 1 if g.board.turn == chess.WHITE else -1
            g.recs.append((planes, pi, to_play))
            g.board.push(_action_to_move(g.board, int(action)))

            if playable(g.board):
                begin_search(g)
                next_active.append(g)
            else:
                s, st = _finalize_game(g, mat_thresh)
                all_samples.extend(s); all_stats.append(st)
                # Refill to keep batches full.
                ng = spawn()
                if ng is not None and playable(ng.board):
                    begin_search(ng)
                    next_active.append(ng)
        active = next_active

    return all_samples, all_stats
