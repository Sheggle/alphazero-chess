"""Plain UCT Monte-Carlo Tree Search — no neural network.

Classic four phases per simulation:
  1. Selection   — descend by UCB1 until we hit a node with unexpanded moves.
  2. Expansion   — add one new child.
  3. Simulation  — random rollout from that child to a terminal state.
  4. Backprop    — propagate the outcome up the path.

Perspective bookkeeping (the part that's easy to get wrong): a rollout returns
`z`, the outcome from player +1's view (+1/0/-1). A node stores its value sum
`w` *from the perspective of the player who moved into it* (i.e. its parent's
`to_play`). That way a parent comparing its children with UCB1 reads each
child's mean value directly as "how good is this move for me". For the edge
parent -> child the mover is `parent.to_play`, so the child accumulates
`z * parent.to_play`.

This works for any game object exposing: `to_play` (+1/-1), `legal_moves()`,
`apply(action)`, `is_terminal()`, and `result()` (+1's perspective).
"""

from __future__ import annotations

import math
import random


class Node:
    __slots__ = ("state", "parent", "to_play", "children", "untried", "n", "w")

    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.to_play = state.to_play
        self.children: dict[int, "Node"] = {}
        self.untried: list[int] = state.legal_moves()
        self.n = 0          # visit count
        self.w = 0.0        # value sum, in the mover-into-this-node's perspective

    @property
    def q(self) -> float:
        return self.w / self.n if self.n > 0 else 0.0

    def is_fully_expanded(self) -> bool:
        return not self.untried


class MCTS:
    def __init__(self, n_sims: int = 100, c: float = 1.4, rng: random.Random | None = None):
        self.n_sims = n_sims
        self.c = c  # UCB1 exploration constant (~sqrt(2) is the textbook value)
        self.rng = rng or random.Random()

    def search(self, root_state) -> int:
        """Run simulations from `root_state`, return the most-visited move."""
        root = Node(root_state)
        for _ in range(self.n_sims):
            node = self._select(root)
            z = self._rollout(node.state)
            self._backprop(node, z)
        # Robust choice: the move we explored most, not the highest mean.
        return max(root.children.items(), key=lambda kv: kv[1].n)[0]

    def action_visits(self, root_state) -> dict[int, int]:
        """Like `search` but expose the full visit distribution (for tests)."""
        root = Node(root_state)
        for _ in range(self.n_sims):
            node = self._select(root)
            z = self._rollout(node.state)
            self._backprop(node, z)
        return {a: child.n for a, child in root.children.items()}

    # --- phases ---

    def _select(self, node: Node) -> Node:
        """Descend until we reach a terminal node or expand a new child."""
        while not node.state.is_terminal():
            if not node.is_fully_expanded():
                return self._expand(node)
            node = self._best_uct_child(node)
        return node

    def _expand(self, node: Node) -> Node:
        action = node.untried.pop(self.rng.randrange(len(node.untried)))
        child = Node(node.state.apply(action), parent=node)
        node.children[action] = child
        return child

    def _best_uct_child(self, node: Node) -> Node:
        log_n = math.log(node.n)
        best, best_score = None, -math.inf
        for child in node.children.values():
            # child.q is already in `node`'s perspective (good-for-mover).
            score = child.q + self.c * math.sqrt(log_n / child.n)
            if score > best_score:
                best, best_score = child, score
        return best

    def _rollout(self, state) -> int:
        """Random playout to the end; return outcome in +1's perspective."""
        while not state.is_terminal():
            moves = state.legal_moves()
            state = state.apply(moves[self.rng.randrange(len(moves))])
        return state.result()

    def _backprop(self, node: Node, z: int) -> None:
        """Walk to the root, crediting each node in its mover's perspective."""
        while node is not None:
            node.n += 1
            parent = node.parent
            if parent is not None:
                # Mover into `node` was parent.to_play.
                node.w += z * parent.to_play
            node = parent
