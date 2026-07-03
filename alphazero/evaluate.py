"""Score an agent against the exact solver — the sharpest progress signal.

`optimal_move_rate` enumerates every reachable non-terminal position and checks
whether the agent's chosen move is game-theoretically optimal. 1.0 means the
agent plays perfectly everywhere it could be asked to move.
"""

from __future__ import annotations

from .solver import optimal_actions
from .tictactoe import TicTacToe


def _all_states(state, seen):
    key = (state.board, state.to_play)
    if key in seen:
        return
    seen[key] = state
    if state.is_terminal():
        return
    for a in state.legal_moves():
        _all_states(state.apply(a), seen)


def all_nonterminal_states() -> list:
    seen: dict = {}
    _all_states(TicTacToe(), seen)
    return [s for s in seen.values() if not s.is_terminal()]


def optimal_move_rate(agent, states=None) -> float:
    """Fraction of positions where the agent picks an optimal move."""
    states = states if states is not None else all_nonterminal_states()
    good = 0
    for s in states:
        if agent.select(s) in optimal_actions(s):
            good += 1
    return good / len(states)
