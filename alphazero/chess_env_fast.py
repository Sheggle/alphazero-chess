"""Fast drop-in for `ChessGame`, backed by the `fastchess` Rust extension.

`FastChessGame` is a behavioral replacement for `alphazero.chess_env.ChessGame`:
it exposes the exact same generic-game interface the search uses
(`action_size`, `to_play`, `legal_moves()`, `apply()`, `is_terminal()`,
`result()`, `ply`) and produces **identical action indices** to `ChessGame`.

The board mechanics (legal move generation, make-move, terminal/result with
draw rules) run in Rust/shakmaty via the `fastchess` extension. Action-index
encoding reuses `encode_move_canonical` from `chess_env`, with the same
canonical mirroring `encode_move` does, so indices match byte-for-byte.

`encode_board` (and anything else that wants a python-chess board) keeps working
because we expose a lazily-reconstructed `.board` property. It is rebuilt from
the fast board's FEN once per state and cached; we override `ep_square` with the
"always" en-passant target so the encoder's ep plane matches the original
`ChessGame` exactly (python-chess sets `ep_square` on any double pawn push).
"""

from __future__ import annotations

import chess

from .chess_env import ACTION_SIZE, encode_move_canonical

# Load the compiled `fastchess` extension. The repo also has a crate directory
# literally named `fastchess/` at the root; when the repo root is on sys.path it
# imports as an empty *namespace* package and shadows the real module. So we put
# the built package dir (fastchess/pybuild) first and verify the real module
# (the one exposing `Board`) actually loaded.
import importlib as _importlib
import pathlib as _pathlib
import sys as _sys

_so_dir = str(_pathlib.Path(__file__).resolve().parent.parent / "fastchess" / "pybuild")
if _so_dir not in _sys.path:
    _sys.path.insert(0, _so_dir)
import fastchess  # noqa: E402

if not hasattr(fastchess, "Board"):  # a namespace-package shadow won the import
    _sys.modules.pop("fastchess", None)
    fastchess = _importlib.import_module("fastchess")
if not hasattr(fastchess, "Board"):  # pragma: no cover
    raise ImportError(
        f"loaded fastchess from {getattr(fastchess, '__file__', '?')} but it has "
        f"no Board class; build it with `cd fastchess && maturin build --release` "
        f"and ensure fastchess/pybuild is importable"
    )


def _ply_from_fen(fen: str) -> int:
    """python-chess `Board.ply()`: 2*(fullmove-1) + (0 white / 1 black)."""
    parts = fen.split()
    fullmove = int(parts[5])
    white = parts[1] == "w"
    return 2 * (fullmove - 1) + (0 if white else 1)


class FastChessGame:
    action_size = ACTION_SIZE

    __slots__ = ("_fc", "_ply", "_legal", "_indices", "_index_map", "_board")

    def __init__(self, fc=None, *, fen: str | None = None, ply: int | None = None):
        if fc is None:
            fc = fastchess.Board(fen) if fen is not None else fastchess.Board()
        self._fc = fc
        self._ply = ply if ply is not None else _ply_from_fen(fc.fen())
        self._legal: list[int] | None = None
        self._indices: list[int] | None = None
        self._index_map: dict[int, int] | None = None
        self._board: chess.Board | None = None

    # --- index encoding (identical mapping to chess_env.encode_move) ---

    def _ensure_legal(self):
        if self._indices is not None:
            return
        white = self._fc.turn_white()
        tuples = self._fc.legal_tuples()  # (from, to, promo) in real-board coords
        indices: list[int] = []
        for f, t, p in tuples:
            if white:
                ff, tt = f, t
            else:  # mirror into the canonical White-up frame (square ^ 56 flips rank)
                ff, tt = f ^ 56, t ^ 56
            indices.append(encode_move_canonical(ff, tt, p if p else None))
        self._legal = tuples
        self._indices = indices
        self._index_map = {a: j for j, a in enumerate(indices)}

    # --- generic game interface ---

    @property
    def to_play(self) -> int:
        return 1 if self._fc.turn_white() else -1

    def legal_moves(self) -> list[int]:
        self._ensure_legal()
        return self._indices

    def apply(self, action: int) -> "FastChessGame":
        self._ensure_legal()
        j = self._index_map.get(action)
        if j is None:
            raise ValueError(f"action {action} not legal in this position")
        new_fc = self._fc.apply_index_copy(j)
        return FastChessGame(new_fc, ply=self._ply + 1)

    def is_terminal(self) -> bool:
        return self._fc.is_terminal()

    def result(self) -> int:
        """Outcome from White (+1) perspective: +1 win, -1 loss, 0 draw."""
        return self._fc.result()

    @property
    def ply(self) -> int:
        return self._ply

    # --- fast encode / material (Rust, no python-chess) ---

    def encode(self):
        """(18,8,8) float32 planes, built in Rust directly from the board.

        Bit-for-bit identical to `chess_encode.encode_board(self)`, but without
        reconstructing a python-chess board or walking its piece_map in Python.
        """
        return self._fc.encode()

    def material_diff(self) -> int:
        """White-perspective material balance (P1 N3 B3 R5 Q9), in Rust."""
        return self._fc.material_diff()

    # --- python-chess view for encode_board / tactics (lazy, cached) ---

    @property
    def board(self) -> chess.Board:
        if self._board is None:
            b = chess.Board(self._fc.fen())
            # python-chess sets ep_square on any double push; mirror that so the
            # encoder's ep plane matches ChessGame exactly.
            ep = self._fc.ep_square()
            b.ep_square = ep
            self._board = b
        return self._board

    def __str__(self) -> str:
        return str(self.board)
