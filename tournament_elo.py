"""Round-robin between training checkpoints -> Bradley-Terry Elo -> JSON.

Each checkpoint plays every other checkpoint G games (colors alternated). A few
opening plies are sampled from the policy (per-game rng) for variety, then moves
come from Gumbel-MCTS search. Outputs tournament_results.json (players + Elo +
W/N matrices); plot_elo.py turns it into the chart.

Run on the box:  PYTHONPATH=.:fastchess/pybuild python tournament_elo.py
"""
import json, sys, time, random
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "fastchess" / "pybuild"))
from alphazero.chess_net import ChessNet, ChessEvaluator
from alphazero.chess_env import ChessGame
from alphazero.gumbel import GumbelMCTS
import chess

# (label, checkpoint, frames in millions)  -- frames for iter<110 estimated at ~92k/iter
PLAYERS = [
    ("i50", "models/chess_gpu/iter_00050.pt", 4.6),
    ("i100", "models/chess_gpu/iter_00100.pt", 9.2),
    ("i150", "models/chess_gpu/iter_00150.pt", 14.5),
    ("i250", "models/chess_gpu/iter_00250.pt", 25.2),
    ("i350", "models/chess_gpu/iter_00350.pt", 35.6),
    ("i450", "models/chess_gpu/iter_00450.pt", 46.0),
    ("i550", "models/chess_gpu/iter_00550.pt", 56.4),
    ("i600", "models/chess_gpu/iter_00600.pt", 61.5),
]
G_PER_PAIR = 8        # games per pair (colors alternated)
SIMS = 16
MC = 16
MAX_PLY = 160
OPEN_PLIES = 4        # sampled-from-policy opening plies for variety
DEV = "cuda" if torch.cuda.is_available() else "cpu"

PVAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def material_white(b):
    return sum(PVAL.get(p.piece_type, 0) * (1 if p.color == chess.WHITE else -1)
               for p in b.piece_map().values())


def load_ev(path):
    ck = torch.load(ROOT / path, map_location=DEV)
    net = ChessNet(ck["channels"], ck["blocks"]).to(DEV).eval()
    net.load_state_dict(ck["state_dict"])
    return ChessEvaluator(net, device=DEV)


def play_game(ev_w, ev_b, rng):
    g = ChessGame()
    while not g.is_terminal() and g.ply < MAX_PLY:
        ev = ev_w if g.to_play == 1 else ev_b
        if g.ply < OPEN_PLIES:
            probs, _ = ev.predict(g)
            legal = g.legal_moves()
            p = probs[legal].astype(np.float64); s = p.sum()
            a = int(rng.choice(legal, p=p / s)) if s > 0 else int(rng.choice(legal))
        else:
            a, _ = GumbelMCTS(ev, n_sims=SIMS, max_considered=MC, c_scale=0.3, rng=rng).run(
                g, add_noise=False)
        g = g.apply(int(a))
    if g.is_terminal():
        return g.result()  # white perspective: +1 white win, -1 black, 0 draw
    md = material_white(g.board)
    return 1 if md >= 2 else (-1 if md <= -2 else 0)


def fit_elo(W, N):
    """Bradley-Terry MM (Zermelo). W[i,j]=score of i vs j, N=games. -> Elo, centered."""
    n = len(W); gamma = np.ones(n); wins = W.sum(axis=1)
    for _ in range(2000):
        ng = gamma.copy()
        for i in range(n):
            denom = sum(N[i, j] / (gamma[i] + gamma[j]) for j in range(n) if j != i and N[i, j] > 0)
            if denom > 0 and wins[i] > 0:
                ng[i] = wins[i] / denom
        ng = np.clip(ng, 1e-9, None); ng /= np.exp(np.mean(np.log(ng)))
        if np.max(np.abs(ng - gamma)) < 1e-9:
            gamma = ng; break
        gamma = ng
    R = 400 * np.log10(gamma); return R - R.mean()


def main():
    n = len(PLAYERS)
    evs = [load_ev(p[1]) for p in PLAYERS]
    print(f"{n} players, {DEV}, {G_PER_PAIR} games/pair, sims={SIMS}", flush=True)
    W = np.zeros((n, n)); N = np.zeros((n, n))
    t0 = time.time(); gid = 0; total = n * (n - 1) // 2 * G_PER_PAIR
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(G_PER_PAIR):
                rng = np.random.default_rng(1000 * i + 100 * j + k)
                if k % 2 == 0:
                    r = play_game(evs[i], evs[j], rng)        # i white
                    si = 1.0 if r > 0 else (0.0 if r < 0 else 0.5)
                else:
                    r = play_game(evs[j], evs[i], rng)        # j white -> flip
                    si = 1.0 if r < 0 else (0.0 if r > 0 else 0.5)
                W[i, j] += si; W[j, i] += 1 - si; N[i, j] += 1; N[j, i] += 1
                gid += 1
            print(f"  {PLAYERS[i][0]} vs {PLAYERS[j][0]}: score {W[i,j]:.1f}/{N[i,j]:.0f} "
                  f"| {gid}/{total} games, {time.time()-t0:.0f}s", flush=True)
    elo = fit_elo(W, N)
    out = {"players": [{"label": PLAYERS[k][0], "frames_M": PLAYERS[k][2],
                        "elo": round(float(elo[k]), 1), "score": round(float(W[k].sum()), 1),
                        "games": int(N[k].sum())} for k in range(n)],
           "W": W.tolist(), "N": N.tolist(), "sims": SIMS, "g_per_pair": G_PER_PAIR}
    Path(ROOT / "tournament_results.json").write_text(json.dumps(out, indent=2))
    print("\n=== ELO (centered) ===", flush=True)
    for pl in sorted(out["players"], key=lambda x: x["frames_M"]):
        print(f"  {pl['label']:5s} {pl['frames_M']:5.1f}M frames -> {pl['elo']:+7.1f} Elo "
              f"({pl['score']:.0f}/{pl['games']})", flush=True)
    print(f"\nspan: {max(p['elo'] for p in out['players']) - min(p['elo'] for p in out['players']):.0f} Elo",
          flush=True)


if __name__ == "__main__":
    main()
