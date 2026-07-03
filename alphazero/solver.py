"""Exact minimax solver for tic-tac-toe — the ground truth for everything else.

TTT is small enough to solve completely. We use this to:
  - confirm perfect play is a draw,
  - label the optimal value/best-actions of any position,
  - score an agent's move choices against optimal.

`solve(state)` returns the game-theoretic value from `state.to_play`'s
perspective: +1 = the side to move can force a win, 0 = draw, -1 = loss.
Results are memoized over the whole reachable game tree (~5478 states).
"""

from __future__ import annotations

from functools import lru_cache

from .tictactoe import TicTacToe


@lru_cache(maxsize=None)
def solve(state: TicTacToe) -> int:
    """Game-theoretic value from the perspective of the player to move."""
    if state.is_terminal():
        # result() is from +1's perspective; flip to side-to-move's view.
        return state.result() * state.to_play
    best = -2
    for a in state.legal_moves():
        # Child is from the opponent's perspective, so negate.
        v = -solve(state.apply(a))
        if v > best:
            best = v
    return best


def optimal_actions(state: TicTacToe) -> list[int]:
    """All moves that achieve the game-theoretic value."""
    best = solve(state)
    return [a for a in state.legal_moves() if -solve(state.apply(a)) == best]
