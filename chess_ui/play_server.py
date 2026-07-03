"""Local server to play against the trained AlphaZero chess net at 2048 sims / L=16
(the production operating point), using the Rust leaf-parallel PUCT search (fastchess).

Run from the repo root:  PYTHONPATH=. uv run python chess_ui/play_server.py
Then open http://localhost:8088
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import chess
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fastchess" / "pybuild"))
import fastchess  # noqa: E402
from alphazero.chess_net import ChessNet  # noqa: E402
from alphazero.chess_env import ChessGame  # noqa: E402
from alphazero.chess_encode import encode_board  # noqa: E402
from chess_ui.repetition_wrapper import position_key, choose_move  # noqa: E402

SIMS, L, C_PUCT = 2048, 16, 1.5  # production operating point

CKPT = ROOT / "models/chess_gpu/playnet.pt"
if not CKPT.exists():
    CKPT = ROOT / "models/chess_keeper/play_net.pt"
_ck = torch.load(CKPT, map_location="cpu", weights_only=False)
_net = ChessNet(_ck["channels"], _ck["blocks"])
_net.load_state_dict(_ck["state_dict"])
_net.eval()
_DEV = "mps" if torch.backends.mps.is_available() else "cpu"
_net = _net.to(_DEV)
_HTML = (Path(__file__).resolve().parent / "index.html").read_text()


@torch.no_grad()
def _eval_fn(planes):  # (M,18,8,8) f32 -> (full logits (M,4672) f32, values (M,) f32)
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


# Per-game position history (position_key list) for the repetition-aware wrapper.
# Keyed by an optional client game_id; a fresh start position resets that game's slot.
_HISTORY: dict[str, list[str]] = {}


def net_move(fen: str, sims: int, game_id: str = "_default"):
    """(uci, value) from leaf-parallel PUCT search at `sims`/L=16, or (None, None) if over.

    Applies the repetition-aware wrapper: if the search's move revisits a prior position
    while winning, play the best non-repeating move instead (see repetition_wrapper.py).
    """
    board = chess.Board(fen)
    if board.is_game_over():
        return None, None
    if board == chess.Board():                 # new game from the start position
        _HISTORY[game_id] = []
    hist = _HISTORY.setdefault(game_id, [])
    visited = set(hist)                         # every position the game has passed through
    _action, uci, value = fastchess.search_position(_eval_fn, fen, int(sims), L, C_PUCT)
    uci = choose_move(board, uci, value, visited, _value_batch)
    # Record both the position moved FROM and the resulting position, so a later move
    # that returns the game to either parity is recognised as a repetition.
    hist.append(position_key(board))
    after = board.copy(); after.push(chess.Move.from_uci(uci)); hist.append(position_key(after))
    return uci, value


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, _HTML, "text/html")
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        if self.path == "/api/move":
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or "{}")
            uci, value = net_move(req["fen"], int(req.get("sims", SIMS)),
                                  str(req.get("game_id", "_default")))
            self._send(200, json.dumps({"move": uci, "value": value}))
        else:
            self._send(404, "{}")

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8088
    print(f"Chess UI ready → http://localhost:{port}   (net: {CKPT.name}, {SIMS} sims / L={L} on {_DEV})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
