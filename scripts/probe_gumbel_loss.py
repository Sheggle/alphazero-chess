"""Root-cause probe for the 'more sims -> more losses to perfect' anomaly.

Loads ttt_gumbel_3sim.pt, builds GumbelAgents at several eval-sim counts, and:
  1. reproduces losses vs PerfectAgent over multiple seeds,
  2. instruments the final action selection on every agent move: compares the
     agent's pick vs the sequential-halving most-visited action vs completed-Q
     argmax vs raw-policy argmax, and flags blunders against the solver,
  3. tests whether 'most-visited' selection removes the blunders / the anomaly,
  4. sweeps max_considered at fixed sims.
"""
import sys
import math
import random
import numpy as np
import torch

from alphazero.net import TicTacToeNet, NetEvaluator
from alphazero.gumbel import GumbelMCTS
from alphazero.agents import PerfectAgent
from alphazero.arena import play_match
from alphazero.solver import optimal_actions, solve
from alphazero.tictactoe import TicTacToe


def load_eval(path="models/ttt_gumbel_3sim.pt"):
    ck = torch.load(path, weights_only=False)
    net = TicTacToeNet(channels=ck["channels"])
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return NetEvaluator(net, device="cpu")


EVAL = load_eval()


class InstrumentedGumbel(GumbelMCTS):
    """Expose, for one root, the candidate scores + alternative selections."""

    def analyze(self, root_state):
        from alphazero.az_mcts import AZNode
        root = AZNode(root_state)
        root_value = self._expand(root)
        root.n = 1
        root.w = root_value
        legal = root_state.legal_moves()
        logits = self._root_logits(root, legal)
        gumbel = np.zeros(len(legal))  # eval: no noise
        m = min(self.max_considered, len(legal), max(2, self.n_sims))
        order = np.argsort(-(gumbel + logits))
        considered = [legal[i] for i in order[:m]]
        gpref = {legal[i]: gumbel[i] + logits[i] for i in range(len(legal))}
        self._sequential_halving(root, considered, gpref)

        # agent's actual pick
        agent_pick = max(considered, key=lambda a: gpref[a] + self._sigma(root, self._q(root, a)))
        # most-visited among considered (sequential-halving winner-ish)
        most_visited = max(considered, key=lambda a: root.children[a].n)
        # completed-Q policy argmax
        improved = self._completed_policy(root, root_state, root_value, logits, legal)
        compq_pick = int(np.asarray(improved).argmax())
        # raw policy argmax
        raw_pick = int(np.exp(logits - logits.max()).argmax())
        raw_pick = legal[raw_pick]

        info = {
            "agent_pick": agent_pick,
            "most_visited": most_visited,
            "compq_pick": compq_pick,
            "raw_pick": raw_pick,
            "considered": considered,
            "visits": {a: root.children[a].n for a in considered},
            "q": {a: self._q(root, a) for a in considered},
            "sigma": {a: self._sigma(root, self._q(root, a)) for a in considered},
            "logit": {legal[i]: logits[i] for i in range(len(legal))},
            "max_visit": max((c.n for c in root.children.values()), default=0),
        }
        return agent_pick, info


def mk(sims, max_considered=8, seed=0):
    return InstrumentedGumbel(EVAL, n_sims=sims, max_considered=max_considered,
                              rng=np.random.default_rng(seed))


# ---------------------------------------------------------------------------
# 1. Reproduce losses vs perfect across seeds
# ---------------------------------------------------------------------------
def reproduce():
    print("=== 1. Losses vs PerfectAgent (mean over 6 seeds, 200 games) ===")
    for sims in (3, 8, 16, 32):
        losses = []
        for seed in range(6):
            ag = type("A", (), {})()
            mcts = mk(sims, seed=seed)
            ag.select = lambda s, _m=mcts: int(_m.run(s, add_noise=False)[0])
            perfect = PerfectAgent(random.Random(1000 + seed))
            r = play_match(ag, perfect, n_games=200)
            losses.append(r.losses)
        print(f"  sims={sims:2d}: mean losses/200 = {np.mean(losses):5.2f}  (per-seed {losses})")


# ---------------------------------------------------------------------------
# 2 & 3. Trace losing games at 32 sims, compare selection rules
# ---------------------------------------------------------------------------
def trace_and_compare(sims=32, max_considered=8):
    print(f"\n=== 2/3. Trace losses at sims={sims}, max_considered={max_considered} ===")
    blunder_records = []  # (state, agent_move, optimal, mv_move, compq_move, info)
    n_games = 0
    n_losses = 0
    # agent pick vs most-visited disagreement / blunder tallies over ALL agent moves
    tally = {"moves": 0, "agent_blunder": 0, "mv_blunder": 0, "compq_blunder": 0,
             "agent_ne_mv": 0, "mv_fixes_agent_blunder": 0,
             "agent_blunder_and_mv_ok": 0}

    for seed in range(6):
        inst = mk(sims, max_considered, seed=seed)
        perfect = PerfectAgent(random.Random(1000 + seed))
        for i in range(200):
            a_is_x = (i % 2 == 0)
            state = TicTacToe()
            history = []
            while not state.is_terminal():
                agent_turn = (state.to_play == 1 and a_is_x) or (state.to_play == -1 and not a_is_x)
                if agent_turn:
                    pick, info = inst.analyze(state)
                    opt = optimal_actions(state)
                    tally["moves"] += 1
                    ab = pick not in opt
                    mvb = info["most_visited"] not in opt
                    cqb = info["compq_pick"] not in opt
                    tally["agent_blunder"] += ab
                    tally["mv_blunder"] += mvb
                    tally["compq_blunder"] += cqb
                    if pick != info["most_visited"]:
                        tally["agent_ne_mv"] += 1
                    if ab and not mvb:
                        tally["mv_fixes_agent_blunder"] += 1
                        tally["agent_blunder_and_mv_ok"] += 1
                    history.append((state, pick, opt, info))
                    state = state.apply(pick)
                else:
                    state = state.apply(perfect.select(state))
            outcome = state.result()
            a_outcome = outcome if a_is_x else -outcome
            n_games += 1
            if a_outcome < 0:
                n_losses += 1
                # find the value-changing blunders in this lost game
                for (st, mv, opt, info) in history:
                    v_before = solve(st)
                    v_after = -solve(st.apply(mv))
                    if v_after < v_before:  # this move worsened the gt value
                        blunder_records.append((st, mv, opt, v_before, v_after, info))

    print(f"  games={n_games} losses={n_losses}")
    print(f"  agent moves total: {tally['moves']}")
    print(f"  agent blunders (non-optimal pick): {tally['agent_blunder']}")
    print(f"  most-visited blunders:             {tally['mv_blunder']}")
    print(f"  completed-Q argmax blunders:       {tally['compq_blunder']}")
    print(f"  agent_pick != most_visited:        {tally['agent_ne_mv']}")
    print(f"  cases where agent blunders but most-visited is optimal: {tally['mv_fixes_agent_blunder']}")

    print(f"\n  -- value-worsening blunders in LOST games ({len(blunder_records)}) --")
    shown = 0
    for (st, mv, opt, vb, va, info) in blunder_records:
        if shown >= 8:
            break
        shown += 1
        print(f"\n  board (to_play={st.to_play}):")
        print("   " + str(st).replace("\n", "\n   "))
        print(f"   agent played {mv} (value {vb:+d} -> {va:+d}), solver-optimal={sorted(opt)}")
        print(f"   most_visited={info['most_visited']} compq={info['compq_pick']} raw={info['raw_pick']}")
        print(f"   max_visit={info['max_visit']}")
        print(f"   candidate breakdown (action: visits, Q, sigma, logit, g+logit+sigma):")
        for a in info["considered"]:
            score = info["logit"][a] + info["sigma"][a]
            star = " <-- AGENT" if a == mv else ""
            mvtag = " [most-visited]" if a == info["most_visited"] else ""
            optag = " OPT" if a in opt else ""
            print(f"     {a}: n={info['visits'][a]:2d} Q={info['q'][a]:+.3f} "
                  f"sigma={info['sigma'][a]:+.2f} logit={info['logit'][a]:+.3f} "
                  f"score={score:+.3f}{star}{mvtag}{optag}")
    return tally


# ---------------------------------------------------------------------------
# 4. Counterfactual: replay with most-visited selection, measure losses
# ---------------------------------------------------------------------------
def counterfactual_selection():
    print("\n=== 4. Losses vs Perfect using alternative selection rules ===")
    rules = ["agent_pick", "most_visited", "compq_pick"]
    for sims in (3, 8, 16, 32):
        out = {r: [] for r in rules}
        for seed in range(6):
            inst = mk(sims, 8, seed=seed)
            perfect = PerfectAgent(random.Random(1000 + seed))
            counts = {r: 0 for r in rules}
            for i in range(200):
                a_is_x = (i % 2 == 0)
                state = TicTacToe()
                # We must branch the game per rule since picks differ; do 3 replays.
                for r in rules:
                    st = TicTacToe()
                    pf = PerfectAgent(random.Random(1000 + seed + 99991 * (i + 1)))
                    while not st.is_terminal():
                        at = (st.to_play == 1 and a_is_x) or (st.to_play == -1 and not a_is_x)
                        if at:
                            _, info = inst.analyze(st)
                            st = st.apply(info[r])
                        else:
                            st = st.apply(pf.select(st))
                    outcome = st.result()
                    ao = outcome if a_is_x else -outcome
                    if ao < 0:
                        counts[r] += 1
            for r in rules:
                out[r].append(counts[r])
        print(f"  sims={sims:2d}: " + "  ".join(
            f"{r}={np.mean(out[r]):5.2f}" for r in rules))


# ---------------------------------------------------------------------------
# 5. max_considered sweep at 32 sims
# ---------------------------------------------------------------------------
def sweep_max_considered():
    print("\n=== 5. max_considered sweep at sims=32 (agent_pick losses) ===")
    for mc in (2, 4, 8):
        losses = []
        for seed in range(6):
            inst = mk(32, mc, seed=seed)
            ag = type("A", (), {})()
            ag.select = lambda s, _m=inst: int(_m.run(s, add_noise=False)[0])
            perfect = PerfectAgent(random.Random(1000 + seed))
            r = play_match(ag, perfect, n_games=200)
            losses.append(r.losses)
        print(f"  max_considered={mc}: mean losses/200 = {np.mean(losses):5.2f} ({losses})")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "1"):
        reproduce()
    if which in ("all", "2"):
        trace_and_compare(32, 8)
    if which in ("all", "4"):
        counterfactual_selection()
    if which in ("all", "5"):
        sweep_max_considered()
