"""Quantify the net's style + absolute strength using Stockfish as ground truth.

Two things the project never had:
  (1) absolute Elo — bracket the net vs Stockfish skill levels.
  (2) an objective read on the 'sacrifices material for activity' style the user saw —
      log material balance, the net's own value, and Stockfish's eval per ply, and
      measure how much material the net is down while still winning.

Net moves: local fastchess leaf-parallel PUCT @ configurable sims (MPS forward).
Run:  cd repo && PYTHONPATH=.:fastchess/pybuild .venv/bin/python <thisfile> <mode> [args]
"""
import sys, json, time, os
from pathlib import Path
import numpy as np, torch
import chess, chess.engine

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "fastchess" / "pybuild"))
import fastchess
from alphazero.chess_net import ChessNet

SF_PATH = os.environ.get("SF_PATH", "/opt/homebrew/bin/stockfish")
NET = ROOT / "models/chess_gpu/playnet.pt"
L, CPUCT = 16, 1.5
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

_ck = torch.load(NET, map_location="cpu", weights_only=False)
_net = ChessNet(_ck["channels"], _ck["blocks"]); _net.load_state_dict(_ck["state_dict"]); _net.eval(); _net = _net.to(DEV)

@torch.no_grad()
def _eval_fn(planes):
    x = torch.from_numpy(planes).to(DEV); lg, v = _net(x)
    return (np.ascontiguousarray(lg.float().cpu().numpy(), np.float32),
            np.ascontiguousarray(v.float().cpu().numpy(), np.float32))

def net_move(board, sims):
    a, uci, val = fastchess.search_position(_eval_fn, board.fen(), int(sims), L, CPUCT)
    return chess.Move.from_uci(uci), val  # val = net value, side-to-move perspective

def material_white(board):
    v = {chess.PAWN:1, chess.KNIGHT:3, chess.BISHOP:3, chess.ROOK:5, chess.QUEEN:9}
    return sum(v.get(p.piece_type,0)*(1 if p.color==chess.WHITE else -1) for p in board.piece_map().values())

def new_sf():
    return chess.engine.SimpleEngine.popen_uci(SF_PATH)

def sf_cp_white(sf, board, depth=10):
    try:
        sc = sf.analyse(board, chess.engine.Limit(depth=depth))["score"].white()
        return sc.score(mate_score=10000)
    except Exception:
        return None

def play_net_vs_sf(sf, net_white, sf_skill, net_sims, analyse_sf, max_ply=220):
    """Return (result_white in {1,0,-1}, per-ply rows). result from WHITE perspective."""
    board = chess.Board(); rows = []
    sf.configure({"Skill Level": int(sf_skill)})
    while not board.is_game_over(claim_draw=True) and board.ply() < max_ply:
        stm_white = board.turn == chess.WHITE
        cp = sf_cp_white(sf, board, analyse_sf) if analyse_sf else None
        rows.append({"ply": board.ply(), "mat_w": material_white(board), "sf_cp_w": cp})
        if stm_white == net_white:
            mv, val = net_move(board, net_sims)
        else:
            mv = sf.play(board, chess.engine.Limit(time=0.05)).move
        if mv not in board.legal_moves:  # safety
            mv = list(board.legal_moves)[0]
        board.push(mv)
    if board.is_game_over(claim_draw=True):
        o = board.outcome(claim_draw=True)
        rw = 0 if (o is None or o.winner is None) else (1 if o.winner == chess.WHITE else -1)
    else:
        cp = sf_cp_white(sf, board, 12)  # adjudicate unfinished games by Stockfish eval, not material
        rw = 1 if (cp is not None and cp >= 150) else (-1 if (cp is not None and cp <= -150) else 0)
    return rw, rows

def elo_from_score(p, n):
    if p <= 0: return -800.0
    if p >= 1: return 800.0
    return 400*np.log10(p/(1-p))

def bracket(net_sims=512, games_per=6, skills=(0,2,4,6,8,10)):
    """Net vs Stockfish at several skill levels -> where does the net cross 50%?"""
    sf = new_sf()
    out = []
    for sk in skills:
        w=d=l=0; t0=time.time()
        for g in range(games_per):
            netw = (g % 2 == 0)
            rw, _ = play_net_vs_sf(sf, netw, sk, net_sims, analyse_sf=0)
            nz = rw if netw else -rw
            w += nz>0; d += nz==0; l += nz<0
        sc = (w + 0.5*d)/games_per
        print(f"[bracket] SF skill {sk:2d}: net {w}-{d}-{l} = {sc:.2f}  elo_vs_this {elo_from_score(sc,games_per):+.0f}  [{time.time()-t0:.0f}s]", flush=True)
        out.append({"skill": sk, "score": sc, "w": w, "d": d, "l": l})
        (ROOT/"style_bracket.json").write_text(json.dumps(out, indent=2))
    sf.quit()

def style(net_sims=1024, n_games=6, sf_skill=5):
    """Play net vs SF, log material + net-value + SF-eval to characterize the style."""
    sf = new_sf()
    allrows = []
    for g in range(n_games):
        netw = (g % 2 == 0)
        rw, rows = play_net_vs_sf(sf, netw, sf_skill, net_sims, analyse_sf=10)
        for r in rows: r["net_is_white"] = netw; r["game"] = g; r["result_w"] = rw
        allrows += rows
        # net-perspective material: + = net ahead
        won = (rw==1)==netw
        mats = [ (r["mat_w"] if netw else -r["mat_w"]) for r in rows ]
        print(f"[style] game {g}: net_{'W' if netw else 'B'} result_w {rw} netwon={won} "
              f"mat(min/mean/max from net view) {min(mats)}/{np.mean(mats):.1f}/{max(mats)}", flush=True)
    (ROOT/"style_rows.json").write_text(json.dumps(allrows))
    # aggregate: in games the net WON, how much material was it down on average?
    import collections
    net_won_rows = [r for r in allrows if (r["result_w"]==1)==r["net_is_white"] and r["result_w"]!=0]
    if net_won_rows:
        nm = [ (r["mat_w"] if r["net_is_white"] else -r["mat_w"]) for r in net_won_rows ]
        print(f"[style] WON games: net material (net view) mean {np.mean(nm):+.2f}, "
              f"frac plies down>=1 pawn {np.mean([x<=-1 for x in nm]):.2f}, min {min(nm)}", flush=True)
    sf.quit()

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "bracket"
    if mode == "bracket":
        bracket(net_sims=int(sys.argv[2]) if len(sys.argv)>2 else 512)
    elif mode == "style":
        style(net_sims=int(sys.argv[2]) if len(sys.argv)>2 else 1024)
    print("DONE", flush=True)
