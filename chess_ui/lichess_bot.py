"""Minimal Lichess Bot API client driving our AlphaZero net (fastchess leaf-parallel
PUCT, L=16), sims scaled to the clock. Accepts standard blitz/rapid challenges, and
seeks rated blitz vs other online bots so a rating builds. Plays as the BOT account.

  PYTHONPATH=. uv run python chess_ui/lichess_bot.py
  token in /tmp/lichess_bot_token (bot:play scope)
"""
import sys, json, time, threading, random
from pathlib import Path
import requests
import chess
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "fastchess" / "pybuild"))
import fastchess
from alphazero.chess_net import ChessNet
from alphazero.chess_env import ChessGame
from alphazero.chess_encode import encode_board
from chess_ui.repetition_wrapper import position_key, choose_move

TOKEN = Path("/tmp/lichess_bot_token").read_text().strip()
H = {"Authorization": f"Bearer {TOKEN}"}
BASE = "https://lichess.org"
ME = "sheggle-bot"
L, C_PUCT = 16, 1.5
MPS_SIMS_PER_MS = 1.35   # ~2048 sims in ~1.5s on MPS

# ---- net ----
CKPT = ROOT / "models/chess_gpu/playnet.pt"
_ck = torch.load(CKPT, map_location="cpu", weights_only=False)
_net = ChessNet(_ck["channels"], _ck["blocks"]); _net.load_state_dict(_ck["state_dict"]); _net.eval()
_DEV = "mps" if torch.backends.mps.is_available() else "cpu"
_net = _net.to(_DEV)
_lock = threading.Lock()   # serialize MPS access across game threads


@torch.no_grad()
def _eval_fn(planes):
    x = torch.from_numpy(planes).to(_DEV)
    logits, values = _net(x)
    return (np.ascontiguousarray(logits.float().cpu().numpy(), dtype=np.float32),
            np.ascontiguousarray(values.float().cpu().numpy(), dtype=np.float32))


@torch.no_grad()
def _value_batch(boards):
    """Net value (each board's own side-to-move POV) for the wrapper's 1-ply lookahead."""
    planes = np.stack([encode_board(ChessGame(b)) for b in boards]).astype(np.float32)
    _, values = _net(torch.from_numpy(planes).to(_DEV))
    return values.float().cpu().numpy().reshape(-1)


def pick_sims(rem_ms, inc_ms):
    if rem_ms < 15000:                       # low on time -> move fast, don't flag
        budget = min(400, rem_ms * 0.05 + inc_ms * 0.5)
    else:
        budget = min(1200, inc_ms * 0.7 + rem_ms * 0.02)
    return max(48, min(2048, int(budget * MPS_SIMS_PER_MS)))


def play_game(game_id):
    try:
        r = requests.get(f"{BASE}/api/bot/game/stream/{game_id}", headers=H, stream=True, timeout=60)
        my_white = None
        for raw in r.iter_lines():
            if not raw:
                continue
            ev = json.loads(raw)
            if ev["type"] == "gameFull":
                my_white = ev["white"].get("id") == ME
                state = ev["state"]
            elif ev["type"] == "gameState":
                state = ev
            else:
                continue
            if state.get("status", "started") != "started":
                print(f"[{game_id}] over: {state.get('status')}", flush=True); return
            board = chess.Board()
            visited = {position_key(board)}   # every position the game has passed through
            for mv in (state["moves"].split() if state["moves"] else []):
                board.push_uci(mv); visited.add(position_key(board))
            if board.is_game_over() or board.turn != (chess.WHITE if my_white else chess.BLACK):
                continue
            rem = state["wtime"] if my_white else state["btime"]
            inc = state["winc"] if my_white else state["binc"]
            sims = pick_sims(rem, inc)
            t = time.time()
            with _lock:
                _a, uci, val = fastchess.search_position(_eval_fn, board.fen(), sims, L, C_PUCT)
                # Repetition-aware: don't shuffle a won game into a 3-fold (see repetition_wrapper.py).
                uci = choose_move(board, uci, val, visited, _value_batch)
            requests.post(f"{BASE}/api/bot/game/{game_id}/move/{uci}", headers=H, timeout=10)
            print(f"[{game_id}] {uci} @{sims}s v{val:+.2f} ({rem/1000:.0f}s left, {time.time()-t:.1f}s think)", flush=True)
    except Exception as e:
        print(f"[{game_id}] error: {e}", flush=True)


def acceptable(ch):
    return (ch["variant"]["key"] == "standard"
            and ch["speed"] in ("blitz", "rapid")
            and ch["challenger"]["id"] != ME)


def seek_games():
    """When idle, challenge a random online bot to rated 3+2 blitz."""
    while True:
        time.sleep(20)
        try:
            playing = requests.get(f"{BASE}/api/account/playing", headers=H, timeout=10).json()
            if playing.get("nowPlaying"):
                continue
            # target the human-calibrated Maia ladder for a meaningful, level rating
            # (avoid the Leela-ODDS bots @2000-3340 and other extreme engines).
            PREFERRED = ["maia1", "maia5", "maia9"]
            online = {json.loads(l)["username"] for l in
                      requests.get(f"{BASE}/api/bot/online?nb=100", headers=H, timeout=10).text.splitlines() if l.strip()}
            avail = [b for b in PREFERRED if b in online and b != ME]
            if not avail:
                continue
            opp = random.choice(avail)
            resp = requests.post(f"{BASE}/api/challenge/{opp}", headers=H, timeout=10,
                                 data={"rated": "true", "clock.limit": 180, "clock.increment": 2,
                                       "color": "random", "variant": "standard"})
            print(f"[seek] challenged {opp} 3+2 rated -> {resp.status_code}", flush=True)
        except Exception as e:
            print(f"[seek] {e}", flush=True)


def main():
    print(f"sheggle-bot online ({CKPT.name} on {_DEV}); streaming events...", flush=True)
    threading.Thread(target=seek_games, daemon=True).start()
    r = requests.get(f"{BASE}/api/stream/event", headers=H, stream=True)
    for raw in r.iter_lines():
        if not raw:
            continue
        ev = json.loads(raw)
        if ev["type"] == "challenge":
            ch = ev["challenge"]
            if acceptable(ch):
                code = requests.post(f"{BASE}/api/challenge/{ch['id']}/accept", headers=H, timeout=10).status_code
                print(f"[challenge] accept {ch['id']} from {ch['challenger']['id']} ({ch['speed']}) -> {code}", flush=True)
            else:
                requests.post(f"{BASE}/api/challenge/{ch['id']}/decline", headers=H, timeout=10)
        elif ev["type"] == "gameStart":
            threading.Thread(target=play_game, args=(ev["game"]["gameId"],), daemon=True).start()


if __name__ == "__main__":
    main()
