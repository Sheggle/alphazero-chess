"""Find the game(s) where 100-sim MCTS loses to random and locate the blunder
by checking each MCTS move against the exact solver."""
import random
from alphazero.agents import MCTSAgent, RandomAgent
from alphazero.solver import optimal_actions, solve
from alphazero.tictactoe import TicTacToe

seed = 0
rng_mcts = random.Random(seed)
rng_rand = random.Random(seed + 1)
mcts = MCTSAgent(n_sims=100, rng=rng_mcts)
rand = RandomAgent(rng=rng_rand)

# Replay matching arena's alternation/seed consumption exactly.
for i in range(200):
    a_is_x = (i % 2 == 0)
    x, o = (mcts, rand) if a_is_x else (rand, mcts)
    state = TicTacToe()
    history = []  # (mover_name, action, was_mcts, value_before, opt_before)
    while not state.is_terminal():
        is_mcts_turn = (state.to_play == 1 and x is mcts) or (state.to_play == -1 and o is mcts)
        if is_mcts_turn:
            v = solve(state)
            opt = optimal_actions(state)
        agent = x if state.to_play == 1 else o
        a = agent.select(state)
        if is_mcts_turn:
            history.append((a, v, opt, state))
        state = state.apply(a)
    outcome = state.result()
    a_outcome = outcome if a_is_x else -outcome
    if a_outcome < 0:  # MCTS lost
        print(f"=== LOSS in game {i} (mcts is {'X' if a_is_x else 'O'}) ===")
        for (a, v_before, opt, st) in history:
            blunder = a not in opt
            tag = "  <-- BLUNDER" if blunder else ""
            print(f"\nstate (to_play={st.to_play}, value={v_before:+d}):")
            print(st)
            print(f"MCTS played {a}; optimal={sorted(opt)}{tag}")
