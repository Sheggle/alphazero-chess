"""Gumbel AlphaZero search — strong and stable at *very* low simulation counts.

Based on Danihelka et al., "Policy improvement by planning with Gumbel" (2022).
Plain AlphaZero acts and trains from MCTS *visit counts*, which carry almost no
information when you only run 2-3 simulations. Gumbel fixes both ends:

  * Acting: at the root, sample `m` candidate actions via Gumbel-top-k over the
    policy logits, then run Sequential Halving — repeatedly give the surviving
    candidates more simulations and drop the worst half, scoring by
    `g + logits + sigma(Q)`. The final survivor is a guaranteed policy
    improvement over the prior, even with a tiny budget.

  * Training target: the *completed-Q* improved policy
    `pi' = softmax(logits + sigma(completedQ))`, where visited actions use their
    search value Q and unvisited actions use a value-completion `v_mix`. This is
    a much stronger, lower-variance target than visit counts at few sims.

Below the root we reuse ordinary PUCT (`AZMCTS._select_child` / `_expand`).
Values follow the same perspective convention as `az_mcts`: a node stores value
in its own to-move perspective, so a root action's value for the root player is
`-child.q`.
"""

from __future__ import annotations

import math

import numpy as np

from .az_mcts import AZMCTS, AZNode


class GumbelMCTS(AZMCTS):
    def __init__(
        self,
        evaluator,
        n_sims: int = 32,
        max_considered: int = 8,
        c_visit: float = 50.0,
        c_scale: float = 1.0,
        c_puct: float = 1.5,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(evaluator, n_sims=n_sims, c_puct=c_puct, rng=rng)
        self.max_considered = max_considered
        self.c_visit = c_visit
        self.c_scale = c_scale

    def run(self, root_state, add_noise: bool = True):
        """Return (chosen_action, improved_policy) for `root_state`.

        `add_noise=True` draws Gumbel noise (exploration, for self-play);
        `add_noise=False` zeroes it for deterministic best-move play (eval).
        """
        root = AZNode(root_state)
        root_value = self._expand(root)
        root.n = 1
        root.w = root_value

        legal = root_state.legal_moves()
        logits = self._root_logits(root, legal)            # prior logits per legal action
        gumbel = (self.rng.gumbel(size=len(legal)) if add_noise
                  else np.zeros(len(legal)))

        # Root candidate width. NOTE: this is min'd with max(2, n_sims), so a
        # larger sim budget *widens* the candidate set. That couples width to
        # sims, which is a trap: the value head is only calibrated on the moves
        # actually expanded during self-play (the top-`width` policy moves at the
        # self-play sim budget). Evaluating the same net at more sims widens the
        # set and queries the value head out-of-distribution on low-policy moves
        # it over-rates -> blunders vs a strong opponent. Keep `max_considered`
        # <= the self-play sim budget so width is identical at train and eval.
        m = min(self.max_considered, len(legal), max(2, self.n_sims))
        order = np.argsort(-(gumbel + logits))             # Gumbel-top-k
        considered = [legal[i] for i in order[:m]]
        gpref = {legal[i]: gumbel[i] + logits[i] for i in range(len(legal))}

        self._sequential_halving(root, considered, gpref)

        improved = self._completed_policy(root, root_state, root_value, logits, legal)

        # Final action: best considered action by g + logits + sigma(Q).
        best = max(considered, key=lambda a: gpref[a] + self._sigma(root, self._q(root, a)))
        return best, improved

    # --- Sequential Halving over the considered root actions ---

    def _sequential_halving(self, root, considered, gpref):
        considered = list(considered)
        budget = self.n_sims
        used = 0
        num_phases = max(1, math.ceil(math.log2(len(considered)))) if len(considered) > 1 else 1

        while used < budget and len(considered) >= 1:
            per = max(1, (budget // num_phases) // len(considered)) if len(considered) > 1 else budget
            for a in considered:
                for _ in range(per):
                    if used >= budget:
                        break
                    self._simulate(root, a)
                    used += 1
                if used >= budget:
                    break
            if len(considered) <= 1 or used >= budget:
                break
            # Keep the better half by g + logits + sigma(Q).
            considered.sort(key=lambda a: gpref[a] + self._sigma(root, self._q(root, a)), reverse=True)
            considered = considered[: max(1, len(considered) // 2)]

    def _simulate(self, root, action):
        """One simulation forced through root->action, PUCT below the child."""
        child = root.children[action]
        path = [child]
        node = child
        while node.is_expanded and not node.state.is_terminal():
            _, node = self._select_child(node)
            path.append(node)
        leaf = path[-1]
        if leaf.state.is_terminal():
            value = leaf.state.result() * leaf.to_play
        else:
            value = self._expand(leaf)
        v = value
        for nd in reversed(path):
            nd.n += 1
            nd.w += v
            v = -v
        root.n += 1  # keep root visit count consistent for sigma's max-visit term

    # --- value / policy helpers ---

    def _q(self, root, action) -> float:
        """Search value of a root action, in the *root* player's perspective."""
        child = root.children[action]
        return -child.q if child.n > 0 else 0.0

    def _sigma(self, root, q: float) -> float:
        max_visit = max((c.n for c in root.children.values()), default=0)
        return (self.c_visit + max_visit) * self.c_scale * q

    def _root_logits(self, root, legal) -> np.ndarray:
        """Recover prior logits over legal actions from child priors (log P)."""
        priors = np.array([root.children[a].prior for a in legal], dtype=np.float64)
        priors = np.clip(priors, 1e-12, 1.0)
        return np.log(priors)

    def _completed_policy(self, root, root_state, root_value, logits, legal) -> np.ndarray:
        """pi' = softmax(logits + sigma(completedQ)) over legal actions, then
        scattered to a full action-size vector."""
        priors = np.exp(logits - logits.max())
        priors /= priors.sum()

        visited = [(i, a) for i, a in enumerate(legal) if root.children[a].n > 0]
        n_total = sum(root.children[a].n for a in legal)
        if visited:
            sum_p = sum(priors[i] for i, _ in visited)
            weighted_q = sum(priors[i] * self._q(root, a) for i, a in visited) / max(sum_p, 1e-12)
            v_mix = (root_value + n_total * weighted_q) / (1 + n_total)
        else:
            v_mix = root_value

        completed_q = np.empty(len(legal))
        for i, a in enumerate(legal):
            completed_q[i] = self._q(root, a) if root.children[a].n > 0 else v_mix

        score = logits + np.array([self._sigma(root, q) for q in completed_q])
        score -= score.max()
        ex = np.exp(score)
        probs = ex / ex.sum()

        pi = np.zeros(root_state.action_size, dtype=np.float32)
        for i, a in enumerate(legal):
            pi[a] = probs[i]
        return pi
