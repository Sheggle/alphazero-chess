"""Self-play game generation for AlphaZero training.

Each game is played by AZMCTS against itself, with Dirichlet noise at the root
and a temperature schedule (sample early for exploration, go greedy later).
For every move we record:

    (state, pi, to_play)

where `pi` is the MCTS visit-count policy (the policy target). After the game
ends we backfill the value target `z` for each sample: the game result seen from
that sample's side to move (+1 won / 0 draw / -1 lost).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .az_mcts import AZMCTS, policy_from_visits
from .gumbel import GumbelMCTS
from .tictactoe import TicTacToe


@dataclass
class Sample:
    state: object
    pi: np.ndarray   # policy target over all actions
    z: float         # value target, side-to-move perspective (filled at game end)


def play_selfplay_game(
    evaluator,
    n_sims: int = 100,
    c_puct: float = 1.5,
    temp_moves: int = 4,
    rng: np.random.Generator | None = None,
    game_factory=TicTacToe,
) -> list[Sample]:
    rng = rng or np.random.default_rng()
    mcts = AZMCTS(evaluator, n_sims=n_sims, c_puct=c_puct, rng=rng)
    state = game_factory()
    records: list[tuple[object, np.ndarray, int]] = []
    move_num = 0

    while not state.is_terminal():
        visits = mcts.run(state, add_noise=True)
        # Training target is the visit distribution (temperature 1).
        pi = policy_from_visits(visits, state.action_size, temperature=1.0)
        records.append((state, pi, state.to_play))

        # Move selection: explore early (temp=1 sample), exploit later (greedy).
        temperature = 1.0 if move_num < temp_moves else 0.0
        move_pi = policy_from_visits(visits, state.action_size, temperature=temperature)
        action = int(rng.choice(state.action_size, p=move_pi))
        state = state.apply(action)
        move_num += 1

    result = state.result()  # +1's perspective
    samples = []
    for st, pi, to_play in records:
        z = result * to_play  # convert to that state's side-to-move perspective
        samples.append(Sample(state=st, pi=pi, z=float(z)))
    return samples


def play_selfplay_game_gumbel(
    evaluator,
    n_sims: int = 8,
    max_considered: int = 8,
    c_visit: float = 50.0,
    c_scale: float = 1.0,
    c_puct: float = 1.5,
    rng: np.random.Generator | None = None,
    game_factory=TicTacToe,
) -> list[Sample]:
    """Self-play with Gumbel search. The move is the Gumbel-selected action and
    the policy target is the completed-Q improved policy (both from one search).
    Gumbel noise supplies exploration, so no temperature schedule is needed."""
    rng = rng or np.random.default_rng()
    state = game_factory()
    records: list[tuple[object, np.ndarray, int]] = []

    while not state.is_terminal():
        mcts = GumbelMCTS(evaluator, n_sims=n_sims, max_considered=max_considered,
                          c_visit=c_visit, c_scale=c_scale, c_puct=c_puct, rng=rng)
        action, improved_pi = mcts.run(state, add_noise=True)
        records.append((state, improved_pi, state.to_play))
        state = state.apply(int(action))

    result = state.result()
    return [Sample(state=st, pi=pi, z=float(result * to_play))
            for st, pi, to_play in records]
