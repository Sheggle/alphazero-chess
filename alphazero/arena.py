"""Play agents against each other and tally results.

`play_match` runs `n_games`, alternating who moves first so neither agent gets a
systematic first-move advantage. Results are reported from agent A's view.
"""

from __future__ import annotations

from dataclasses import dataclass

from .tictactoe import TicTacToe


@dataclass
class MatchResult:
    wins: int = 0      # agent A wins
    draws: int = 0
    losses: int = 0    # agent A losses (B wins)
    games: int = 0

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def score(self) -> float:
        """Win=1, draw=0.5, loss=0, averaged — the usual head-to-head score."""
        return (self.wins + 0.5 * self.draws) / self.games if self.games else 0.0

    def __str__(self) -> str:
        return (
            f"{self.games} games | A: {self.wins}W {self.draws}D {self.losses}L "
            f"| win%={100*self.win_rate:.1f} score={self.score:.3f}"
        )


def play_game(agent_x, agent_o, game=None) -> int:
    """Play one game; agent_x is player +1, agent_o is player -1.

    Returns the result in +1's perspective (+1 / 0 / -1).
    """
    state = game or TicTacToe()
    while not state.is_terminal():
        agent = agent_x if state.to_play == 1 else agent_o
        state = state.apply(agent.select(state))
    return state.result()


def play_match(agent_a, agent_b, n_games: int = 200, game_factory=TicTacToe) -> MatchResult:
    """Alternate sides each game; tally from agent A's perspective."""
    res = MatchResult()
    for i in range(n_games):
        a_is_x = (i % 2 == 0)
        x, o = (agent_a, agent_b) if a_is_x else (agent_b, agent_a)
        outcome = play_game(x, o, game_factory())  # +1's perspective
        a_outcome = outcome if a_is_x else -outcome
        res.games += 1
        if a_outcome > 0:
            res.wins += 1
        elif a_outcome < 0:
            res.losses += 1
        else:
            res.draws += 1
    return res
