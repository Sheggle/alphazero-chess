"""Validate AZMCTS plumbing with a solver-backed oracle evaluator.
If perspective/PUCT is correct, oracle-guided AZMCTS must play optimally."""
import numpy as np
from alphazero.az_mcts import AZMCTS, policy_from_visits
from alphazero.solver import optimal_actions, solve
from alphazero.tictactoe import TicTacToe


class OracleEvaluator:
    """Returns the true solved value (to-move perspective) and a policy that
    puts most mass on optimal moves (so priors are informative but not cheating
    on search)."""
    def predict(self, state):
        if state.is_terminal():
            return np.ones(9) / 9, float(state.result() * state.to_play)
        value = float(solve(state))  # already in to-move perspective
        opt = set(optimal_actions(state))
        probs = np.zeros(9)
        for a in state.legal_moves():
            probs[a] = 1.0 if a in opt else 0.1
        probs /= probs.sum()
        return probs, value


oracle = OracleEvaluator()

# Test a bunch of positions: AZMCTS choice must be optimal per solver.
def az_move(state, n_sims=50, seed=0):
    mcts = AZMCTS(oracle, n_sims=n_sims, rng=np.random.default_rng(seed))
    visits = mcts.run(state)
    pi = policy_from_visits(visits, 9, temperature=0.0)
    return int(pi.argmax())

# 1. corner opening -> must take center
s = TicTacToe().apply(2)
m = az_move(s)
print(f"corner opening: AZMCTS plays {m}, optimal={optimal_actions(s)} -> {'OK' if m in optimal_actions(s) else 'FAIL'}")

# 2. immediate win
s = TicTacToe()
for a in [0,3,1,4]:
    s = s.apply(a)
m = az_move(s)
print(f"immediate win: AZMCTS plays {m}, optimal={optimal_actions(s)} -> {'OK' if m in optimal_actions(s) else 'FAIL'}")

# 3. must block
s = TicTacToe()
for a in [0,4,1]:
    s = s.apply(a)
m = az_move(s)
print(f"must block: AZMCTS plays {m}, optimal={optimal_actions(s)} -> {'OK' if m in optimal_actions(s) else 'FAIL'}")

# 4. full game: oracle-AZMCTS vs oracle-AZMCTS should draw, and never make a
# losing move from a drawn position.
fails = 0
for game_seed in range(30):
    s = TicTacToe()
    moves = 0
    while not s.is_terminal():
        v_before = solve(s)
        m = az_move(s, n_sims=40, seed=game_seed*100+moves)
        if m not in optimal_actions(s):
            fails += 1
        s = s.apply(m)
        moves += 1
    # self-play of optimal players must end in a draw
    if s.result() != 0:
        print(f"  game {game_seed}: NON-DRAW result={s.result()}")
print(f"self-play optimality: {fails} sub-optimal moves across 30 games (want 0)")
