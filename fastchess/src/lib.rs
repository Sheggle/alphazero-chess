//! fastchess: a minimal pyo3 wrapper around `shakmaty` exposing exactly what
//! AlphaZero self-play needs from a chess board, with semantics matched to
//! python-chess (`is_game_over(claim_draw=True)` / `outcome(claim_draw=True)`).
//!
//! UCI strings and (from, to, promo) tuples are produced by hand so they line up
//! byte-for-byte with python-chess (standard castling -> e1g1/e1c1, etc.).

use numpy::{PyArray1, PyArray3, PyArray4, PyArrayMethods};
use pyo3::prelude::*;
use shakmaty::zobrist::{Zobrist64, ZobristHash};
use shakmaty::{
    CastlingMode, CastlingSide, Color, EnPassantMode, Move, Position, Role, Square,
};
use shakmaty::fen::Fen;

mod arena;
mod selfplay;

const N: u8 = 2; // knight  (python-chess piece-type ints)
const B: u8 = 3; // bishop
const R: u8 = 4; // rook
const Q: u8 = 5; // queen

/// Encoded input planes: (18, 8, 8) f32, C-contiguous. Layout MUST match
/// `alphazero/chess_encode.py::encode_board` exactly (canonical / side-to-move
/// frame, White moving up the board):
///   0..5   : side-to-move's pieces  (P,N,B,R,Q,K)
///   6..11  : opponent's pieces      (P,N,B,R,Q,K)
///   12..15 : castling rights (my K, my Q, opp K, opp Q)  -- full plane of 1s
///   16     : en-passant target square
///   17     : halfmove clock, min(clock,100)/100 (full plane)
const N_PLANES: usize = 18;
const PLANE: usize = 64;
const BUF: usize = N_PLANES * PLANE; // 1152

#[inline]
fn fill_plane(buf: &mut [f32], plane: usize, v: f32) {
    let base = plane * PLANE;
    for x in &mut buf[base..base + PLANE] {
        *x = v;
    }
}

/// Piece value for material counting (python-chess: P1 N3 B3 R5 Q9, K=0).
#[inline]
fn role_value(r: Role) -> i32 {
    match r {
        Role::Pawn => 1,
        Role::Knight => 3,
        Role::Bishop => 3,
        Role::Rook => 5,
        Role::Queen => 9,
        Role::King => 0,
    }
}

#[inline]
fn role_to_int(r: Role) -> u8 {
    match r {
        Role::Pawn => 1,
        Role::Knight => 2,
        Role::Bishop => 3,
        Role::Rook => 4,
        Role::Queen => 5,
        Role::King => 6,
    }
}

#[inline]
fn promo_letter(r: Role) -> char {
    match r {
        Role::Knight => 'n',
        Role::Bishop => 'b',
        Role::Rook => 'r',
        Role::Queen => 'q',
        _ => '?',
    }
}

#[inline]
fn sq_index(s: Square) -> u8 {
    u8::from(s)
}

#[inline]
fn push_sq_name(out: &mut String, idx: u8) {
    let file = (b'a' + (idx & 7)) as char;
    let rank = (b'1' + (idx >> 3)) as char;
    out.push(file);
    out.push(rank);
}

/// Build the python-chess-compatible UCI string for a move.
pub(crate) fn move_uci(m: &Move) -> String {
    let from = sq_index(m.from().expect("chess move always has a from-square"));
    let mut out = String::with_capacity(5);
    if let Some(side) = m.castling_side() {
        // Standard-chess castling: king moves two squares toward the rook.
        let rank = from >> 3; // king's rank
        let to_file = match side {
            CastlingSide::KingSide => 6u8,  // g-file
            CastlingSide::QueenSide => 2u8, // c-file
        };
        push_sq_name(&mut out, from);
        push_sq_name(&mut out, rank * 8 + to_file);
        return out;
    }
    push_sq_name(&mut out, from);
    push_sq_name(&mut out, sq_index(m.to()));
    if let Some(p) = m.promotion() {
        out.push(promo_letter(p));
    }
    out
}

/// (from, to, promo) tuple matching python-chess Move fields.
/// `to` is the *king destination* for castling (g/c file), promo is 0 or 2..5.
fn move_tuple(m: &Move) -> (u8, u8, u8) {
    let from = sq_index(m.from().unwrap());
    if let Some(side) = m.castling_side() {
        let rank = from >> 3;
        let to_file = match side {
            CastlingSide::KingSide => 6u8,
            CastlingSide::QueenSide => 2u8,
        };
        return (from, rank * 8 + to_file, 0);
    }
    let promo = m.promotion().map(role_to_int).unwrap_or(0);
    (from, sq_index(m.to()), promo)
}

#[inline]
fn zkey(pos: &shakmaty::Chess) -> u64 {
    // EnPassantMode::Legal mirrors python-chess's _transposition_key, which only
    // distinguishes an ep square when a legal ep capture exists.
    u64::from(pos.zobrist_hash::<Zobrist64>(EnPassantMode::Legal))
}

#[inline]
fn castling_sig(pos: &shakmaty::Chess) -> u64 {
    u64::from(pos.castles().castling_rights())
}

#[pyclass]
pub struct Board {
    pos: shakmaty::Chess,
    /// Zobrist keys for the current reversible window (since the last
    /// irreversible move), including the current position. Used for
    /// threefold/fivefold detection à la python-chess.
    rep: Vec<u64>,
}

impl Board {
    fn fresh(pos: shakmaty::Chess) -> Self {
        let k = zkey(&pos);
        Board { pos, rep: vec![k] }
    }

    /// count of the current position's key within the reversible window.
    #[inline]
    fn cur_reps(&self) -> u32 {
        let cur = *self.rep.last().unwrap();
        self.rep.iter().filter(|&&k| k == cur).count() as u32
    }

    /// max repetition count of any key in the window (cheap, window is short).
    fn max_reps(&self) -> u32 {
        let mut best = 0u32;
        for (i, k) in self.rep.iter().enumerate() {
            let c = self.rep[i..].iter().filter(|&&x| x == *k).count() as u32;
            if c > best {
                best = c;
            }
        }
        best
    }

    /// occurrences of `key` in the reversible window.
    #[inline]
    fn count_key(&self, key: u64) -> u32 {
        self.rep.iter().filter(|&&k| k == key).count() as u32
    }

    fn is_fifty(&self) -> bool {
        self.pos.halfmoves() >= 100
    }
    fn is_seventyfive(&self) -> bool {
        self.pos.halfmoves() >= 150
    }

    /// Fill `buf` (length BUF, assumed pre-zeroed) with the canonical encoding.
    /// Bit-for-bit identical to `encode_board`: when it's Black to move we mirror
    /// (vertical rank flip `sq ^ 56` + colour swap) so the side to move is always
    /// "White moving up", exactly like python-chess `board.mirror()`.
    fn encode_into(&self, buf: &mut [f32]) {
        let white = self.pos.turn() == Color::White;
        let board = self.pos.board();
        for sq in board.occupied() {
            let p = board.piece_at(sq).unwrap();
            let idx = u8::from(sq) as usize; // rank*8 + file
            let f = idx & 7; // file is unchanged by a vertical mirror
            let (r, base) = if white {
                let base = if p.color == Color::White { 0 } else { 6 };
                (idx >> 3, base)
            } else {
                // mirror: rank flips, colour swaps
                let base = if p.color == Color::White { 6 } else { 0 };
                ((idx >> 3) ^ 7, base)
            };
            let role_idx = (role_to_int(p.role) - 1) as usize; // P=0..K=5
            buf[(base + role_idx) * PLANE + r * 8 + f] = 1.0;
        }

        // Castling: planes[12..15] = (stm_K, stm_Q, opp_K, opp_Q). Under the
        // colour-swapping mirror, the side to move's rights map to plane 12/13
        // regardless of real colour (mirrored-White == original side-to-move).
        let cr = self.pos.castles();
        let (stm, opp) = if white {
            (Color::White, Color::Black)
        } else {
            (Color::Black, Color::White)
        };
        if cr.has(stm, CastlingSide::KingSide) {
            fill_plane(buf, 12, 1.0);
        }
        if cr.has(stm, CastlingSide::QueenSide) {
            fill_plane(buf, 13, 1.0);
        }
        if cr.has(opp, CastlingSide::KingSide) {
            fill_plane(buf, 14, 1.0);
        }
        if cr.has(opp, CastlingSide::QueenSide) {
            fill_plane(buf, 15, 1.0);
        }

        // En-passant (python-chess "always" semantics: set on any double push),
        // mirrored together with the board when Black is to move.
        if let Some(ep) = self.pos.ep_square(EnPassantMode::Always) {
            let e = u8::from(ep) as usize;
            let f = e & 7;
            let r = if white { e >> 3 } else { (e >> 3) ^ 7 };
            buf[16 * PLANE + r * 8 + f] = 1.0;
        }

        // Halfmove clock plane (turn-independent).
        let hm = (self.pos.halfmoves().min(100) as f32) / 100.0;
        if hm != 0.0 {
            fill_plane(buf, 17, hm);
        }
    }
}

#[pymethods]
impl Board {
    #[new]
    #[pyo3(signature = (fen=None))]
    fn new(fen: Option<&str>) -> PyResult<Self> {
        let pos: shakmaty::Chess = match fen {
            None => shakmaty::Chess::default(),
            Some(s) => {
                let f: Fen = s
                    .parse()
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("bad FEN: {e}")))?;
                f.into_position(CastlingMode::Standard).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("illegal position: {e}"))
                })?
            }
        };
        Ok(Board::fresh(pos))
    }

    fn clone_board(&self) -> Board {
        Board {
            pos: self.pos.clone(),
            rep: self.rep.clone(),
        }
    }

    fn fen(&self) -> String {
        Fen::from_position(self.pos.clone(), EnPassantMode::Legal).to_string()
    }

    fn turn_white(&self) -> bool {
        self.pos.turn() == Color::White
    }

    fn halfmove_clock(&self) -> u32 {
        self.pos.halfmoves()
    }

    /// Legal moves as python-chess-compatible UCI strings.
    fn legal_uci(&self) -> Vec<String> {
        self.pos.legal_moves().iter().map(move_uci).collect()
    }

    /// Legal moves as (from, to, promo) tuples.
    fn legal_tuples(&self) -> Vec<(u8, u8, u8)> {
        self.pos.legal_moves().iter().map(move_tuple).collect()
    }

    fn num_legal(&self) -> usize {
        self.pos.legal_moves().len()
    }

    /// Play the i-th legal move in place (fast path; index from legal_* order).
    fn apply_index(&mut self, i: usize) -> PyResult<()> {
        let moves = self.pos.legal_moves();
        let m = moves
            .get(i)
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("legal move index out of range"))?
            .clone();
        self.play(&m);
        Ok(())
    }

    /// Play a move given by python-chess UCI (matches our own UCI formatting).
    fn apply_uci(&mut self, uci: &str) -> PyResult<()> {
        let m = self
            .pos
            .legal_moves()
            .iter()
            .find(|m| move_uci(m) == uci)
            .cloned()
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("no legal move {uci}"))
            })?;
        self.play(&m);
        Ok(())
    }

    /// Return a *new* Board with the i-th legal move applied (drop-in for
    /// ChessGame.apply, which builds a fresh game each ply).
    fn apply_index_copy(&self, i: usize) -> PyResult<Board> {
        let mut b = self.clone_board();
        b.apply_index(i)?;
        Ok(b)
    }

    // ---- terminal / result, matched to python-chess claim_draw=True ----

    fn is_checkmate(&self) -> bool {
        self.pos.is_checkmate()
    }

    fn is_terminal(&self) -> bool {
        if self.pos.is_checkmate() || self.pos.is_stalemate() {
            return true;
        }
        if self.pos.is_insufficient_material() {
            return true;
        }
        if self.is_seventyfive() {
            return true;
        }
        let reps = self.cur_reps();
        if reps >= 5 {
            return true; // fivefold (automatic)
        }
        if reps >= 3 {
            return true; // claimable threefold
        }
        if self.is_fifty() {
            return true; // claimable fifty-move
        }
        // Claimable-by-announcing-a-move edge cases. python-chess can claim a
        // threefold if SOME legal move reaches a key already seen >= 2 times in
        // the window (not necessarily the current key). Guard on whether any
        // key in the window repeats at all.
        if self.max_reps() >= 2 {
            for m in self.pos.legal_moves().iter() {
                let nk = zkey(&self.pos.clone().play(m).unwrap());
                if self.count_key(nk) >= 2 {
                    return true;
                }
            }
        }
        if self.pos.halfmoves() >= 99 {
            // fifty by announcing a non-zeroing move that reaches clock 100
            for m in self.pos.legal_moves().iter() {
                if m.is_zeroing() {
                    continue;
                }
                if self.pos.clone().play(m).unwrap().halfmoves() >= 100 {
                    return true;
                }
            }
        }
        false
    }

    /// Outcome from White's perspective: +1 White win, -1 Black win, 0 draw.
    /// Only checkmate yields a non-draw under claim_draw semantics.
    fn result(&self) -> i32 {
        if self.pos.is_checkmate() {
            // side to move is mated -> loses
            if self.pos.turn() == Color::White {
                -1
            } else {
                1
            }
        } else {
            0
        }
    }

    // ---- encoding helpers (bonus, for a full ChessGame replacement) ----

    /// (square, is_white, role_int) for every piece.
    fn piece_map(&self) -> Vec<(u8, bool, u8)> {
        let board = self.pos.board();
        let mut out = Vec::with_capacity(32);
        for sq in board.occupied() {
            let p = board.piece_at(sq).unwrap();
            out.push((sq_index(sq), p.color == Color::White, role_to_int(p.role)));
        }
        out
    }

    /// En-passant target square (python-chess style: set on any double push).
    fn ep_square(&self) -> Option<u8> {
        self.pos.ep_square(EnPassantMode::Always).map(sq_index)
    }

    /// (white_kingside, white_queenside, black_kingside, black_queenside)
    fn castling_rights(&self) -> (bool, bool, bool, bool) {
        let cr = self.pos.castles();
        (
            cr.has(Color::White, CastlingSide::KingSide),
            cr.has(Color::White, CastlingSide::QueenSide),
            cr.has(Color::Black, CastlingSide::KingSide),
            cr.has(Color::Black, CastlingSide::QueenSide),
        )
    }

    /// (18,8,8) f32 input planes, canonical/side-to-move, matching encode_board.
    /// Returned as an owned, writable, C-contiguous numpy array (zero per-element
    /// torch work on the Python side; one buffer, one H2D copy later).
    fn encode<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f32>>> {
        let mut buf = vec![0f32; BUF];
        self.encode_into(&mut buf);
        Ok(PyArray1::from_vec_bound(py, buf).reshape([N_PLANES, 8, 8])?)
    }

    /// Material balance from White's perspective (P1 N3 B3 R5 Q9), like
    /// `chess_train.material_diff`. Lets the self-play loop avoid python-chess.
    fn material_diff(&self) -> i32 {
        let board = self.pos.board();
        let mut d = 0;
        for sq in board.occupied() {
            let p = board.piece_at(sq).unwrap();
            let v = role_value(p.role);
            d += if p.color == Color::White { v } else { -v };
        }
        d
    }

    /// Is `sq` attacked by `by_white`'s pieces?
    fn is_attacked(&self, sq: u8, by_white: bool) -> bool {
        let s = Square::new(sq as u32);
        let color = if by_white { Color::White } else { Color::Black };
        let board = self.pos.board();
        !(board.attacks_to(s, color, board.occupied())).is_empty()
    }
}

impl Board {
    fn play(&mut self, m: &Move) {
        let prev_castle = castling_sig(&self.pos);
        self.pos.play_unchecked(m);
        let irreversible = self.pos.halfmoves() == 0 || castling_sig(&self.pos) != prev_castle;
        if irreversible {
            self.rep.clear();
        }
        self.rep.push(zkey(&self.pos));
    }
}

// ---- AlphaZero move-index encoding (canonical White-up frame) ----
// Mirrors `alphazero/chess_env.py::encode_move_canonical` exactly. Used by the
// in-crate self-play engine so action indices match python-chess byte-for-byte.

const QUEEN_DIRS: [(i32, i32); 8] =
    [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)];
const KNIGHT_DELTAS: [(i32, i32); 8] =
    [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)];

#[inline]
fn isign(x: i32) -> i32 {
    (x > 0) as i32 - (x < 0) as i32
}

/// Encode a move expressed in the canonical (White-up) frame into [0,4672).
/// `promo` is a python-chess piece-type int (0=none, 2=N, 3=B, 4=R, 5=Q).
fn encode_move_canonical(from_sq: i32, to_sq: i32, promo: u8) -> i32 {
    let (ff, fr) = (from_sq & 7, from_sq >> 3);
    let (tf, tr) = (to_sq & 7, to_sq >> 3);
    let (df, dr) = (tf - ff, tr - fr);

    // Underpromotion (N/B/R). Queen promotions fall through to queen slides.
    let under = match promo {
        2 => Some(0),
        3 => Some(1),
        4 => Some(2),
        _ => None,
    };
    if let Some(u) = under {
        let plane = 64 + u * 3 + (df + 1);
        return from_sq * 73 + plane;
    }
    // Knight move?
    let (adf, adr) = (df.abs(), dr.abs());
    if (adf == 1 && adr == 2) || (adf == 2 && adr == 1) {
        let idx = KNIGHT_DELTAS.iter().position(|&d| d == (df, dr)).unwrap() as i32;
        return from_sq * 73 + 56 + idx;
    }
    // Queen slide (straight or diagonal).
    let dir = (isign(df), isign(dr));
    let dist = adf.max(adr);
    let pidx = QUEEN_DIRS.iter().position(|&d| d == dir).unwrap() as i32;
    from_sq * 73 + pidx * 7 + (dist - 1)
}

// ---- in-crate (pub(crate)) helpers for the self-play engine ----
impl Board {
    /// Construct a fresh Board from a FEN (full 6-field). Used by the arena to
    /// start games from a fixed opening suite. History resets at the FEN.
    pub(crate) fn from_fen(fen: &str) -> PyResult<Board> {
        let f: Fen = fen
            .parse()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("bad FEN: {e}")))?;
        let pos = f.into_position(CastlingMode::Standard).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("illegal position: {e}"))
        })?;
        Ok(Board::fresh(pos))
    }

    pub(crate) fn dup(&self) -> Board {
        Board {
            pos: self.pos.clone(),
            rep: self.rep.clone(),
        }
    }

    pub(crate) fn legal_moves_vec(&self) -> Vec<Move> {
        self.pos.legal_moves().iter().cloned().collect()
    }

    /// Canonical AlphaZero action index for a legal move on this board, matching
    /// `chess_env.encode_move` (mirror to White-up frame when Black to move).
    pub(crate) fn action_index(&self, m: &Move) -> i32 {
        let (f, t, p) = move_tuple(m);
        let (f, t) = if self.pos.turn() == Color::White {
            (f as i32, t as i32)
        } else {
            ((f ^ 56) as i32, (t ^ 56) as i32) // square_mirror = flip rank
        };
        encode_move_canonical(f, t, p)
    }

    pub(crate) fn play_move(&mut self, m: &Move) {
        self.play(m);
    }

    pub(crate) fn to_play_i8(&self) -> i8 {
        if self.pos.turn() == Color::White {
            1
        } else {
            -1
        }
    }

    pub(crate) fn is_terminal_b(&self) -> bool {
        self.is_terminal()
    }
    pub(crate) fn result_white(&self) -> i32 {
        self.result()
    }
    pub(crate) fn material_white(&self) -> i32 {
        self.material_diff()
    }
    pub(crate) fn encode_buf(&self, buf: &mut [f32]) {
        for x in buf.iter_mut() {
            *x = 0.0;
        }
        self.encode_into(buf);
    }
    pub(crate) fn start_pos() -> Board {
        Board::fresh(shakmaty::Chess::default())
    }
}

/// Batch-encode a list of boards into one (N,18,8,8) f32 numpy array, amortizing
/// the FFI crossing. One allocation, one return; identical layout to `encode`.
#[pyfunction]
fn encode_batch<'py>(
    py: Python<'py>,
    boards: Vec<PyRef<'py, Board>>,
) -> PyResult<Bound<'py, PyArray4<f32>>> {
    let n = boards.len();
    let mut buf = vec![0f32; n * BUF];
    for (i, b) in boards.iter().enumerate() {
        b.encode_into(&mut buf[i * BUF..(i + 1) * BUF]);
    }
    Ok(PyArray1::from_vec_bound(py, buf).reshape([n, N_PLANES, 8, 8])?)
}

#[pymodule]
fn fastchess(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Board>()?;
    m.add_function(wrap_pyfunction!(encode_batch, m)?)?;
    m.add_function(wrap_pyfunction!(selfplay::run_selfplay, m)?)?;
    m.add_function(wrap_pyfunction!(arena::arena_match, m)?)?;
    m.add_function(wrap_pyfunction!(arena::arena_match_openings, m)?)?;
    m.add_function(wrap_pyfunction!(arena::arena_bench, m)?)?;
    m.add_function(wrap_pyfunction!(arena::search_position, m)?)?;
    let _ = (N, B, R, Q);
    Ok(())
}
