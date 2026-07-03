//! Score-Four (3D Connect 4, 4x4x4) alpha-beta engine — Rust port of
//! `alphazero/connect4_ab.py`. Pure classic game AI: bitboard + incremental
//! 76-line heuristic + iterative-deepening PVS alpha-beta. NO net, NO learning.
//!
//! Separate from the chess paths in this crate (own module + exports only).
//!
//! Representation (identical semantics to the Python engine):
//!   * two u64 bitboards, bb[0] = player +1 discs, bb[1] = player -1 discs;
//!     bit `cell` set iff occupied. `cell = col*4 + z = x*16 + y*4 + z`, so the
//!     flat cell index equals the env's `_cell` and a column index is an env
//!     action.
//!   * `h[16]` column heights, `cnt[side][line]` bead counts per line so the
//!     line heuristic is maintained incrementally on make/undo (touch only the
//!     ~7 lines through the placed cell) and read in O(1) at a leaf.
//!
//! Heuristic (side +1 frame): a pure line with k of our beads (k=1..3) adds
//! `W[k]`; a pure opponent line subtracts `W[k]`; mixed/empty lines are 0.
//! W[3] is normalised to 1.0; W1,W2 are passed in. Values are f64 and summed in
//! the same per-line order as Python for a bit-identical static eval.
//!
//! Search: iterative deepening + aspiration windows; principal-variation search
//! (null-window re-search); TT-move + killers + history + child-heuristic move
//! ordering; immediate-win short-circuit; mate-distance pruning; fixed-size TT
//! with depth-preferred replacement.
//!
//! This module is PURE Rust (no pyo3, no wall-clock types) so it compiles for
//! both native (pyo3 wrapper in lib.rs) and `wasm32-unknown-unknown` (the
//! `c4wasm` crate). The only platform-specific concern — when to stop the
//! iterative deepening — is injected as a `stop_fn: Fn(nodes) -> bool` closure
//! (native supplies an `Instant` deadline; wasm supplies `Date.now()` or a node
//! budget).

const NLINES: usize = 76;
const MATE: f64 = 1.0e9;
const MATE_TH: f64 = MATE - 1.0e4;
const INF: f64 = 1.0e18;
const EPS: f64 = 1.0e-6; // PVS null-window width (< min eval granularity ~0.096)
const MAXPLY: usize = 66;

const TT_BITS: usize = 21; // 2^21 entries (~48 MB)
const TT_SIZE: usize = 1 << TT_BITS;
const TT_MASK: u64 = (TT_SIZE as u64) - 1;

const F_EXACT: u8 = 0;
const F_LOWER: u8 = 1;
const F_UPPER: u8 = 2;

// ------------------------------------------------------------- line geometry
pub(crate) struct Geo {
    pub(crate) line_cells: Vec<[usize; 4]>, // 76 lines, flat cell indices
    pub(crate) cell_lines: Vec<Vec<usize>>, // per cell (0..63) -> line indices through it
}

fn build_geo() -> Geo {
    // Mirror alphazero/connect4_env.py::_gen_lines EXACTLY (same append order),
    // so the parity static eval sums line contributions in identical order.
    let n: i32 = 4;
    let vals = [-1i32, 0, 1];
    let mut dirs: Vec<(i32, i32, i32)> = Vec::new();
    for &dx in &vals {
        for &dy in &vals {
            for &dz in &vals {
                if !(dx == 0 && dy == 0 && dz == 0) {
                    dirs.push((dx, dy, dz));
                }
            }
        }
    }
    let cell = |x: i32, y: i32, z: i32| -> usize { (x * 16 + y * 4 + z) as usize };
    // BTreeSet (not HashSet) so the core has no dependency on std's RNG-seeded
    // hasher, which can fail on wasm32-unknown-unknown.
    let mut seen: std::collections::BTreeSet<[usize; 4]> = std::collections::BTreeSet::new();
    let mut line_cells: Vec<[usize; 4]> = Vec::new();
    for &(dx, dy, dz) in &dirs {
        for x in 0..n {
            for y in 0..n {
                for z in 0..n {
                    let (px, py, pz) = (x - dx, y - dy, z - dz);
                    if (0..n).contains(&px) && (0..n).contains(&py) && (0..n).contains(&pz) {
                        continue;
                    }
                    let (ex, ey, ez) = (x + 3 * dx, y + 3 * dy, z + 3 * dz);
                    if !((0..n).contains(&ex) && (0..n).contains(&ey) && (0..n).contains(&ez)) {
                        continue;
                    }
                    let cells = [
                        cell(x, y, z),
                        cell(x + dx, y + dy, z + dz),
                        cell(x + 2 * dx, y + 2 * dy, z + 2 * dz),
                        cell(x + 3 * dx, y + 3 * dy, z + 3 * dz),
                    ];
                    let mut key = cells;
                    key.sort_unstable();
                    if seen.contains(&key) {
                        continue;
                    }
                    seen.insert(key);
                    line_cells.push(cells);
                }
            }
        }
    }
    assert_eq!(line_cells.len(), NLINES);
    let mut cell_lines: Vec<Vec<usize>> = vec![Vec::new(); 64];
    for (li, cells) in line_cells.iter().enumerate() {
        for &c in cells.iter() {
            cell_lines[c].push(li);
        }
    }
    Geo { line_cells, cell_lines }
}

pub(crate) fn geo() -> &'static Geo {
    use std::sync::OnceLock;
    static G: OnceLock<Geo> = OnceLock::new();
    G.get_or_init(build_geo)
}

// ------------------------------------------------------------------- TT entry
#[derive(Clone, Copy)]
struct TTEntry {
    key0: u64,
    key1: u64,
    val: f64,
    depth: i16,
    flag: u8,
    best: u8,
    used: bool,
}

impl Default for TTEntry {
    fn default() -> Self {
        TTEntry { key0: 0, key1: 0, val: 0.0, depth: -1, flag: 0, best: 16, used: false }
    }
}

// --------------------------------------------------------------------- engine
pub struct C4 {
    bb: [u64; 2],
    h: [u8; 16],
    cnt: [[u8; NLINES]; 2],
    score: f64,       // player +1 frame
    turn: usize,      // 0 -> +1 to move, 1 -> -1
    wp: [f64; 5],     // [0, W1, W2, W3, 0]
    nodes: u64,
    stop_fn: Box<dyn Fn(u64) -> bool>, // (nodes) -> should stop (deadline/budget)
    stop: bool,
    root_best: i32,
    tt: Vec<TTEntry>,
    killers: [[i32; 2]; MAXPLY],
    history: [u64; 16],
    geo: &'static Geo,
}

impl C4 {
    pub fn new(w1: f64, w2: f64) -> Self {
        C4 {
            bb: [0, 0],
            h: [0; 16],
            cnt: [[0; NLINES]; 2],
            score: 0.0,
            turn: 0,
            wp: [0.0, w1, w2, 1.0, 0.0],
            nodes: 0,
            stop_fn: Box::new(|_| false),
            stop: false,
            root_best: -1,
            tt: vec![TTEntry::default(); TT_SIZE],
            killers: [[-1; 2]; MAXPLY],
            history: [0; 16],
            geo: geo(),
        }
    }

    /// Build an engine and load a position in one step.
    pub fn from_board(board: &[i8], to_play: i8, w1: f64, w2: f64) -> Self {
        let mut e = C4::new(w1, w2);
        e.set_position(board, to_play);
        e
    }

    /// Inject the stop predicate (called ~every 2048 nodes with the node count).
    pub fn set_stop_fn(&mut self, f: Box<dyn Fn(u64) -> bool>) {
        self.stop_fn = f;
    }

    /// Load position from a flat length-64 board (`cell = x*16+y*4+z`), values
    /// in {0,+1,-1}, and side to move (+1/-1).
    fn set_position(&mut self, board: &[i8], to_play: i8) {
        self.bb = [0, 0];
        self.h = [0; 16];
        self.cnt = [[0; NLINES]; 2];
        for cell in 0..64usize {
            match board[cell] {
                1 => self.bb[0] |= 1u64 << cell,
                -1 => self.bb[1] |= 1u64 << cell,
                _ => {}
            }
        }
        for col in 0..16usize {
            let base = col * 4;
            let mut filled = 0u8;
            for z in 0..4 {
                if board[base + z] != 0 {
                    filled = (z + 1) as u8;
                }
            }
            self.h[col] = filled;
        }
        let mut score = 0.0;
        for (li, cells) in self.geo.line_cells.iter().enumerate() {
            let mut a = 0u8;
            let mut b = 0u8;
            for &c in cells.iter() {
                match board[c] {
                    1 => a += 1,
                    -1 => b += 1,
                    _ => {}
                }
            }
            self.cnt[0][li] = a;
            self.cnt[1][li] = b;
            if a > 0 && b == 0 {
                score += self.wp[a as usize];
            } else if b > 0 && a == 0 {
                score -= self.wp[b as usize];
            }
        }
        self.score = score;
        self.turn = if to_play == 1 { 0 } else { 1 };
    }

    #[inline]
    fn make(&mut self, col: usize) {
        let z = self.h[col] as usize;
        let cell = col * 4 + z;
        let lines = &self.geo.cell_lines[cell];
        if self.turn == 0 {
            self.bb[0] |= 1u64 << cell;
            for &li in lines {
                let b = self.cnt[1][li];
                let a = self.cnt[0][li];
                if b == 0 {
                    self.score += self.wp[(a + 1) as usize] - self.wp[a as usize];
                } else if a == 0 {
                    self.score += self.wp[b as usize];
                }
                self.cnt[0][li] = a + 1;
            }
        } else {
            self.bb[1] |= 1u64 << cell;
            for &li in lines {
                let a = self.cnt[0][li];
                let b = self.cnt[1][li];
                if a == 0 {
                    self.score -= self.wp[(b + 1) as usize] - self.wp[b as usize];
                } else if b == 0 {
                    self.score -= self.wp[a as usize];
                }
                self.cnt[1][li] = b + 1;
            }
        }
        self.h[col] = (z + 1) as u8;
        self.turn ^= 1;
    }

    #[inline]
    fn undo(&mut self, col: usize) {
        self.turn ^= 1;
        let z = (self.h[col] - 1) as usize;
        self.h[col] = z as u8;
        let cell = col * 4 + z;
        let lines = &self.geo.cell_lines[cell];
        if self.turn == 0 {
            self.bb[0] &= !(1u64 << cell);
            for &li in lines {
                let b = self.cnt[1][li];
                let a = self.cnt[0][li] - 1;
                self.cnt[0][li] = a;
                if b == 0 {
                    self.score -= self.wp[(a + 1) as usize] - self.wp[a as usize];
                } else if a == 0 {
                    self.score -= self.wp[b as usize];
                }
            }
        } else {
            self.bb[1] &= !(1u64 << cell);
            for &li in lines {
                let a = self.cnt[0][li];
                let b = self.cnt[1][li] - 1;
                self.cnt[1][li] = b;
                if a == 0 {
                    self.score += self.wp[(b + 1) as usize] - self.wp[b as usize];
                } else if b == 0 {
                    self.score += self.wp[a as usize];
                }
            }
        }
    }

    #[inline]
    fn wins(&self, col: usize) -> bool {
        let cell = col * 4 + self.h[col] as usize;
        let side = self.turn;
        for &li in &self.geo.cell_lines[cell] {
            if self.cnt[side][li] == 3 {
                return true;
            }
        }
        false
    }

    #[inline]
    fn tt_index(&self, k0: u64, k1: u64) -> usize {
        let mix = k0
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(k1.wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
        ((mix >> 29) & TT_MASK) as usize
    }

    fn negamax(&mut self, depth: i32, mut alpha: f64, mut beta: f64, ply: usize) -> f64 {
        self.nodes += 1;
        if (self.nodes & 2047) == 0 && (self.stop_fn)(self.nodes) {
            self.stop = true;
        }
        if self.stop {
            return 0.0;
        }

        // mate-distance pruning
        let mating = MATE - ply as f64;
        if mating < beta {
            beta = mating;
            if alpha >= beta {
                return mating;
            }
        }
        let mated = -(MATE - ply as f64);
        if mated > alpha {
            alpha = mated;
            if alpha >= beta {
                return mated;
            }
        }

        // moves
        let mut moves: [usize; 16] = [0; 16];
        let mut nm = 0usize;
        for c in 0..16usize {
            if self.h[c] < 4 {
                moves[nm] = c;
                nm += 1;
            }
        }
        if nm == 0 {
            return 0.0; // full board -> draw
        }

        // immediate win?
        let side = self.turn;
        for i in 0..nm {
            let col = moves[i];
            let cell = col * 4 + self.h[col] as usize;
            for &li in &self.geo.cell_lines[cell] {
                if self.cnt[side][li] == 3 {
                    if ply == 0 {
                        self.root_best = col as i32;
                    }
                    return MATE - ply as f64;
                }
            }
        }

        if depth == 0 {
            return if side == 0 { self.score } else { -self.score };
        }

        // TT probe
        let (k0, k1) = (self.bb[0], self.bb[1]);
        let idx = self.tt_index(k0, k1);
        let mut tt_move: i32 = -1;
        {
            let e = self.tt[idx];
            if e.used && e.key0 == k0 && e.key1 == k1 {
                tt_move = e.best as i32;
                if e.depth as i32 >= depth {
                    let mut v = e.val;
                    if v > MATE_TH {
                        v -= ply as f64;
                    } else if v < -MATE_TH {
                        v += ply as f64;
                    }
                    match e.flag {
                        F_EXACT => return v,
                        F_LOWER => {
                            if v >= beta {
                                return v;
                            }
                        }
                        F_UPPER => {
                            if v <= alpha {
                                return v;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        // move ordering: tt-move, killers, history + child heuristic
        let k0killer = if ply < MAXPLY { self.killers[ply][0] } else { -1 };
        let k1killer = if ply < MAXPLY { self.killers[ply][1] } else { -1 };
        let sgn = if side == 0 { 1.0 } else { -1.0 };
        let mut keys: [f64; 16] = [0.0; 16];
        for i in 0..nm {
            let col = moves[i];
            if col as i32 == tt_move {
                keys[i] = 1.0e30;
                continue;
            }
            // child heuristic (mover frame)
            self.make(col);
            let ch = self.score * sgn;
            self.undo(col);
            let mut key = ch + (self.history[col] as f64) * 1.0e-3;
            if col as i32 == k0killer {
                key += 1.0e9;
            } else if col as i32 == k1killer {
                key += 5.0e8;
            }
            keys[i] = key;
        }
        // insertion sort by key desc (nm <= 16)
        for i in 1..nm {
            let mk = keys[i];
            let mm = moves[i];
            let mut j = i;
            while j > 0 && keys[j - 1] < mk {
                keys[j] = keys[j - 1];
                moves[j] = moves[j - 1];
                j -= 1;
            }
            keys[j] = mk;
            moves[j] = mm;
        }

        let a_orig = alpha;
        let mut best = -INF;
        let mut best_col = moves[0];
        for i in 0..nm {
            let col = moves[i];
            self.make(col);
            let val = if i == 0 {
                -self.negamax(depth - 1, -beta, -alpha, ply + 1)
            } else {
                let mut v = -self.negamax(depth - 1, -alpha - EPS, -alpha, ply + 1);
                if v > alpha && v < beta && !self.stop {
                    v = -self.negamax(depth - 1, -beta, -alpha, ply + 1);
                }
                v
            };
            self.undo(col);
            if self.stop {
                return 0.0;
            }
            if val > best {
                best = val;
                best_col = col;
                if val > alpha {
                    alpha = val;
                }
            }
            if alpha >= beta {
                // quiet-move cutoff: update killers + history
                if ply < MAXPLY && self.killers[ply][0] != col as i32 {
                    self.killers[ply][1] = self.killers[ply][0];
                    self.killers[ply][0] = col as i32;
                }
                self.history[col] += (depth * depth) as u64;
                break;
            }
        }

        if ply == 0 {
            self.root_best = best_col as i32;
        }

        // TT store (depth-preferred replacement)
        let flag = if best <= a_orig {
            F_UPPER
        } else if best >= beta {
            F_LOWER
        } else {
            F_EXACT
        };
        let e = &mut self.tt[idx];
        let same = e.used && e.key0 == k0 && e.key1 == k1;
        if !e.used || same || (e.depth as i32) <= depth {
            let mut store_v = best;
            if store_v > MATE_TH {
                store_v += ply as f64;
            } else if store_v < -MATE_TH {
                store_v -= ply as f64;
            }
            e.key0 = k0;
            e.key1 = k1;
            e.val = store_v;
            e.depth = depth as i16;
            e.flag = flag;
            e.best = best_col as u8;
            e.used = true;
        }
        best
    }

    fn fallback(&self) -> i32 {
        // take a win, else block, else centre-most, else first legal
        for side in [self.turn, self.turn ^ 1] {
            for col in 0..16usize {
                if self.h[col] < 4 {
                    let cell = col * 4 + self.h[col] as usize;
                    for &li in &self.geo.cell_lines[cell] {
                        if self.cnt[side][li] == 3 {
                            return col as i32;
                        }
                    }
                }
            }
        }
        for col in [5usize, 6, 9, 10] {
            if self.h[col] < 4 {
                return col as i32;
            }
        }
        for col in 0..16usize {
            if self.h[col] < 4 {
                return col as i32;
            }
        }
        -1
    }

    /// Fixed-depth search (no time limit). Returns (col, depth, nodes, score_stm).
    pub fn search_depth(&mut self, depth: i32) -> (i32, i32, u64, f64) {
        self.nodes = 0;
        self.stop = false;
        self.root_best = -1;
        self.killers = [[-1; 2]; MAXPLY];
        self.history = [0; 16];
        let legal: Vec<usize> = (0..16).filter(|&c| self.h[c] < 4).collect();
        if legal.is_empty() {
            return (-1, 0, 0, 0.0);
        }
        if legal.len() == 1 {
            return (legal[0] as i32, 0, 0, 0.0);
        }
        // fixed depth ignores the stop predicate (search to completion)
        self.stop_fn = Box::new(|_| false);
        let score = self.negamax(depth, -INF, INF, 0);
        let col = if self.root_best >= 0 { self.root_best } else { self.fallback() };
        (col, depth, self.nodes, score)
    }

    /// Iterative-deepening search until the injected `stop_fn` fires (or the
    /// tree is solved / exhausted). Set `stop_fn` first via `set_stop_fn`.
    /// Returns (col, depth_completed, nodes, score).
    pub fn search_id(&mut self) -> (i32, i32, u64, f64) {
        self.nodes = 0;
        self.stop = false;
        self.killers = [[-1; 2]; MAXPLY];
        self.history = [0; 16];
        let legal: Vec<usize> = (0..16).filter(|&c| self.h[c] < 4).collect();
        if legal.is_empty() {
            return (-1, 0, 0, 0.0);
        }
        if legal.len() == 1 {
            return (legal[0] as i32, 0, 0, 0.0);
        }

        let mut best = self.fallback();
        let mut best_score = 0.0;
        let mut depth_done = 0;
        let empties: i32 = (0..16).map(|c| 4 - self.h[c] as i32).sum();
        let max_depth = empties.min(64);

        let mut prev = 0.0f64;
        let mut depth = 1;
        while depth <= max_depth {
            self.root_best = -1;
            // aspiration windows (only once we have a stable estimate)
            let mut window = 0.5f64;
            let (mut alpha, mut beta) = if depth >= 5 {
                (prev - window, prev + window)
            } else {
                (-INF, INF)
            };
            let mut score;
            loop {
                score = self.negamax(depth, alpha, beta, 0);
                if self.stop {
                    break;
                }
                if score <= alpha {
                    window *= 4.0;
                    alpha = if window > 1.0e6 { -INF } else { score - window };
                } else if score >= beta {
                    window *= 4.0;
                    beta = if window > 1.0e6 { INF } else { score + window };
                } else {
                    break;
                }
            }
            if self.stop {
                break;
            }
            if self.root_best >= 0 {
                best = self.root_best;
                best_score = score;
                depth_done = depth;
            }
            prev = score;
            if score.abs() > MATE_TH {
                break; // solved
            }
            depth += 1;
        }
        (best, depth_done, self.nodes, best_score)
    }

    /// Depth-0 static eval in the side-to-move frame (for parity checks).
    pub fn static_eval(&self) -> f64 {
        if self.turn == 0 {
            self.score
        } else {
            -self.score
        }
    }

    /// Number of empty cells (>= max useful search depth).
    pub fn empties(&self) -> i32 {
        (0..16).map(|c| 4 - self.h[c] as i32).sum()
    }
}
