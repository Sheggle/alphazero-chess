"""AlphaZero-style MCTS: PUCT guided by a network, no random rollouts.

Differences from plain UCT (`mcts.py`):
  - A leaf is evaluated by the network's *value* head, not a random rollout.
  - Children are created with *prior* probabilities from the network's policy.
  - Selection uses PUCT:  score = Q + c_puct * P * sqrt(N_parent) / (1 + N_child).
  - During self-play we mix Dirichlet noise into the root priors for exploration.

Value/perspective: the evaluator returns a value from the perspective of the
player to move at that state. Each node stores its value sum in *its own*
to-move perspective; backup flips sign one level at a time (negamax). A parent
choosing among children therefore wants to *minimise* the child's stored Q, i.e.
it scores `-child.Q` (good-for-me = bad-for-opponent).
"""

from __future__ import annotations

import math

import numpy as np


class AZNode:
    __slots__ = ("state", "to_play", "prior", "children", "n", "w", "is_expanded")

    def __init__(self, state, prior: float = 0.0):
        self.state = state
        self.to_play = state.to_play
        self.prior = prior
        self.children: dict[int, "AZNode"] = {}
        self.n = 0
        self.w = 0.0           # value sum, in this node's own to-move perspective
        self.is_expanded = False

    @property
    def q(self) -> float:
        return self.w / self.n if self.n > 0 else 0.0


class AZMCTS:
    def __init__(
        self,
        evaluator,
        n_sims: int = 100,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_frac: float = 0.25,
        rng: np.random.Generator | None = None,
    ):
        self.evaluator = evaluator
        self.n_sims = n_sims
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_frac = dirichlet_frac
        self.rng = rng or np.random.default_rng()

    def run(self, root_state, add_noise: bool = False) -> dict[int, int]:
        """Run simulations; return {action: visit_count} at the root."""
        root = AZNode(root_state)
        root_value = self._expand(root)
        # Count the root's own network evaluation as one visit. Without this,
        # sqrt(root.n)=0 on the first simulation zeroes the PUCT prior term for
        # every child, so selection degenerates to the lowest-index move and the
        # first 1-2 sims are wasted (and low-sim policy targets get biased).
        root.n = 1
        root.w = root_value
        if add_noise:
            self._add_dirichlet_noise(root)

        for _ in range(self.n_sims):
            path = [root]
            node = root
            # Selection: descend through expanded, non-terminal nodes.
            while node.is_expanded and not node.state.is_terminal():
                action, node = self._select_child(node)
                path.append(node)

            leaf = path[-1]
            if leaf.state.is_terminal():
                # Terminal value from the leaf's to-move perspective.
                value = leaf.state.result() * leaf.to_play
            else:
                value = self._expand(leaf)
            self._backprop(path, value)

        return {a: child.n for a, child in root.children.items()}

    # --- internals ---

    def _expand(self, node: AZNode) -> float:
        """Attach children with net priors; return the net value of `node`."""
        probs, value = self.evaluator.predict(node.state)
        for a in node.state.legal_moves():
            node.children[a] = AZNode(node.state.apply(a), prior=float(probs[a]))
        node.is_expanded = True
        return value

    def _add_dirichlet_noise(self, root: AZNode) -> None:
        actions = list(root.children)
        noise = self.rng.dirichlet([self.dirichlet_alpha] * len(actions))
        f = self.dirichlet_frac
        for a, n in zip(actions, noise):
            child = root.children[a]
            child.prior = (1 - f) * child.prior + f * n

    def _select_child(self, node: AZNode):
        sqrt_n = math.sqrt(node.n)
        best_a, best_child, best_score = None, None, -math.inf
        for a, child in node.children.items():
            # -child.q: child.q is in the opponent's perspective.
            u = self.c_puct * child.prior * sqrt_n / (1 + child.n)
            score = -child.q + u
            if score > best_score:
                best_a, best_child, best_score = a, child, score
        return best_a, best_child

    def _backprop(self, path, value: float) -> None:
        v = value  # in path[-1]'s (leaf's) to-move perspective
        for node in reversed(path):
            node.n += 1
            node.w += v
            v = -v  # flip for the parent one level up


def policy_from_visits(visits: dict[int, int], action_size: int, temperature: float = 1.0) -> np.ndarray:
    """Turn visit counts into a probability vector over all actions.

    temperature -> 0 makes it greedy (all mass on the most-visited move);
    temperature == 1 is proportional to visits (the training target).
    """
    pi = np.zeros(action_size, dtype=np.float32)
    if temperature <= 1e-6:
        best = max(visits, key=visits.get)
        pi[best] = 1.0
        return pi
    counts = np.array([visits.get(a, 0) for a in range(action_size)], dtype=np.float64)
    counts = counts ** (1.0 / temperature)
    pi[:] = counts / counts.sum()
    return pi
