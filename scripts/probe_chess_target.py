"""Standalone diagnostic: inspect the Gumbel completed-Q POLICY TARGET on a
fresh (random-init) chess net over the tactics suite, as a function of the
sigma sharpening knobs (c_visit, c_scale) and candidate width (max_considered).

For each suite position we run one GumbelMCTS search (add_noise like self-play)
and report, averaged over the suite split:
  - target entropy (nats); uniform reference = ln(#legal)
  - top1 / top2 target mass (how peaked the SGD target is)
  - P(solution): probability the target assigns to the correct move
  - sol_in_considered: fraction where the solution is even in the candidate set
  - greedy_solve: fraction the *eval* (no-noise) search actually solves

This isolates the cold-start collapse without any training. Run-only, writes
nothing to the repo source.
"""
import sys, os, json, math
import numpy as np
import torch

from alphazero.chess_env import ChessGame, encode_move
from alphazero.chess_net import ChessNet, ChessEvaluator
from alphazero.gumbel import GumbelMCTS
import chess


def action_to_uci(board, action):
    for mv in board.legal_moves:
        if encode_move(board, mv) == action:
            return mv.uci()
    return None


def probe(net, suite, sims, mc, c_visit, c_scale, seed=0):
    ev = ChessEvaluator(net)
    agg = {}  # type -> list of dict
    for entry in suite:
        typ = entry["type"]
        board = chess.Board(entry["fen"])
        state = ChessGame(board)
        legal = state.legal_moves()
        sols = set(entry["solutions"])
        sol_actions = {a for a in legal if action_to_uci(board, a) in sols}

        # self-play-style target (with noise)
        g = GumbelMCTS(ev, n_sims=sims, max_considered=mc, c_visit=c_visit,
                       c_scale=c_scale, rng=np.random.default_rng(seed))
        _, pi = g.run(state, add_noise=True)
        p = pi[np.array(legal)]
        p = p / p.sum()
        nz = p[p > 1e-9]
        ent = float(-(nz * np.log(nz)).sum())
        order = np.sort(p)[::-1]
        top1 = float(order[0]); top2 = float(order[:2].sum())
        psol = float(sum(pi[a] for a in sol_actions))

        # which actions were considered? reconstruct candidate set deterministically
        # (mirror gumbel.run width logic with no noise for stability of the check)
        # Instead, check via greedy eval whether solved + whether sol could be considered.
        ge = GumbelMCTS(ev, n_sims=sims, max_considered=mc, c_visit=c_visit,
                        c_scale=c_scale, rng=np.random.default_rng(0))
        a_best, _ = ge.run(state, add_noise=False)
        solved = a_best in sol_actions
        # sol-in-considered: top-mc by logits (no-noise considered set)
        # recover logits via prior
        from alphazero.az_mcts import AZNode
        root = AZNode(state); rv = ge._expand(root); root.n = 1; root.w = rv
        logits = ge._root_logits(root, legal)
        m = min(mc, len(legal), max(2, sims))
        considered = set(np.array(legal)[np.argsort(-logits)[:m]].tolist())
        sol_in_cons = len(sol_actions & considered) > 0

        agg.setdefault(typ, []).append(dict(ent=ent, top1=top1, top2=top2,
            psol=psol, nlegal=len(legal), solved=solved, sol_in_cons=sol_in_cons,
            unif=math.log(len(legal))))
    out = {}
    for typ, rows in agg.items():
        out[typ] = {k: round(float(np.mean([r[k] for r in rows])), 3)
                    for k in ("ent", "unif", "top1", "top2", "psol", "nlegal",
                              "solved", "sol_in_cons")}
    return out


def main():
    suite = json.load(open("models/chess/tactics_suite.json"))
    torch.manual_seed(0)
    net = ChessNet(channels=32, blocks=4); net.eval()
    sims = int(os.environ.get("SIMS", 24))
    configs = [
        # (mc, c_visit, c_scale)
        (8, 50.0, 1.0),   # current TTT-ported config
        (8, 50.0, 0.3),
        (8, 50.0, 0.1),
        (8, 10.0, 0.1),
        (16, 50.0, 1.0),
        (16, 50.0, 0.1),
        (24, 50.0, 1.0),
        (24, 50.0, 0.1),
        (24, 10.0, 0.05),
    ]
    print(f"FRESH net CH32/BL4, sims={sims}. uniform entropy ~ ln(nlegal)")
    print(f"{'mc':>3} {'cvis':>5} {'cscl':>5} | {'type':>15} {'ent':>5} {'unif':>5} "
          f"{'top1':>5} {'top2':>5} {'psol':>5} {'solIN':>6} {'solv':>5}")
    for mc, cv, cs in configs:
        res = probe(net, suite, sims, mc, cv, cs)
        for typ in ("mate_in_1", "hanging_capture"):
            r = res[typ]
            print(f"{mc:>3} {cv:>5} {cs:>5} | {typ:>15} {r['ent']:>5} {r['unif']:>5} "
                  f"{r['top1']:>5} {r['top2']:>5} {r['psol']:>5} {r['sol_in_cons']:>6} {r['solved']:>5}")


if __name__ == "__main__":
    main()
