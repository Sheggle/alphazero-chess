"""Agents that pick a move given a state: `select(state) -> action`."""

from __future__ import annotations

import random

import numpy as np

from .az_mcts import AZMCTS, policy_from_visits
from .mcts import MCTS
from .solver import optimal_actions


class RandomAgent:
    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    def select(self, state) -> int:
        moves = state.legal_moves()
        return moves[self.rng.randrange(len(moves))]


class MCTSAgent:
    def __init__(self, n_sims: int = 100, c: float = 1.4, rng: random.Random | None = None):
        self.mcts = MCTS(n_sims=n_sims, c=c, rng=rng or random.Random())

    def select(self, state) -> int:
        return self.mcts.search(state)


class AZAgent:
    """Network + AlphaZero MCTS, greedy (no noise) — for evaluation/play."""

    def __init__(self, evaluator, n_sims: int = 100, c_puct: float = 1.5,
                 rng: np.random.Generator | None = None):
        self.evaluator = evaluator
        self.n_sims = n_sims
        self.c_puct = c_puct
        self.rng = rng or np.random.default_rng()

    def select(self, state) -> int:
        mcts = AZMCTS(self.evaluator, n_sims=self.n_sims, c_puct=self.c_puct, rng=self.rng)
        visits = mcts.run(state, add_noise=False)
        pi = policy_from_visits(visits, state.action_size, temperature=0.0)
        return int(pi.argmax())


class GumbelAgent:
    """Gumbel search, deterministic (no Gumbel noise) — for evaluation/play."""

    def __init__(self, evaluator, n_sims: int = 8, max_considered: int = 3,
                 c_visit: float = 50.0, c_scale: float = 1.0, c_puct: float = 1.5,
                 rng: np.random.Generator | None = None):
        # max_considered defaults to the training candidate width (3 in this
        # project). It must match the width used in self-play, else eval queries
        # the value head out-of-distribution (see gumbel.py / OPTIMIZATION_LOG).
        from .gumbel import GumbelMCTS
        self._mk = lambda: GumbelMCTS(
            evaluator, n_sims=n_sims, max_considered=max_considered,
            c_visit=c_visit, c_scale=c_scale, c_puct=c_puct,
            rng=rng or np.random.default_rng(),
        )

    def select(self, state) -> int:
        action, _ = self._mk().run(state, add_noise=False)
        return int(action)


class RawNetAgent:
    """Greedy over the raw network policy — no search. Tests what the net alone
    has learned."""

    def __init__(self, evaluator):
        self.evaluator = evaluator

    def select(self, state) -> int:
        probs, _ = self.evaluator.predict(state)
        return int(np.asarray(probs).argmax())


class PerfectAgent:
    """Plays a game-theoretically optimal move (uniformly among optimal ones)."""

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    def select(self, state) -> int:
        opt = optimal_actions(state)
        return opt[self.rng.randrange(len(opt))]
