//! Score-Four (3D Connect 4, 4x4x4) alpha-beta engine — pure Rust core (v3).
//!
//! Classic game AI: bitboard + incremental 76-line heuristic + iterative-
//! deepening PVS alpha-beta. NO net, NO learning, NO opening book.
//! Compiles for native (pyo3 wrapper in lib.rs) and wasm32 (`c4wasm`); the only
//! platform concern — when to stop — is injected as `stop_fn(nodes) -> bool`.
//!
//! Representation
//! --------------
//! * two u64 bitboards `bb[0]` (player +1) / `bb[1]` (player -1); bit `cell`
//!   set iff occupied; `cell = col*4 + z = x*16 + y*4 + z` (== env `_cell`).
//! * `h[16]` column heights; `cnt[side][line]` bead counts per line.
//! * `droppable`: u64, one bit per playable drop cell (`col*4 + h[col]`).
//! * `threat[side]`: u64 of cells that complete a 4-line for `side`, maintained
//!   via per-cell refcounts `tcnt`, updated only on the rare transitions of a
//!   line into/out of the "3 own beads, 0 opponent" state. The immediate-win
//!   test is one AND: `threat[stm] & droppable`.
//!
//! Heuristic (integer, +1 frame): pure line with k own beads adds `WP[k]`; pure
//! opponent lines subtract; mixed/empty are 0. Weights scaled x10_000 to i32
//! once at construction — the hot loop is float-free. Scores unscale to floats
//! only at the API edge (`score_to_f64`; mate = +/-(100_000 - plies)).
//!
//! Search: iterative deepening + integer aspiration windows; PVS; null-move
//! pruning (R=2/3, skipped when the mover is under immediate threat); late-move
//! reductions; TT-move + killers + history + O(7)-delta move ordering (NO
//! make/undo during ordering — the delta over lines through the drop cell plus
//! the constant parent score equals the child heuristic exactly); mate-distance
//! pruning; lock-free shared transposition table (16-byte packed entries,
//! 4-way cache-line buckets, XOR-validated atomic pairs, aging) enabling lazy
//! SMP (`search_smp`, native only).

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;

const NLINES: usize = 76;

/// Integer score scale: 1.0 (= W3) == 10_000.
pub const SCALE: i32 = 10_000;
pub const MATE: i32 = 1_000_000_000;
pub const MATE_TH: i32 = 999_000_000;
const INF: i32 = 1_500_000_000;
const MAXPLY: usize = 66;

const F_EXACT: u8 = 1;
const F_LOWER: u8 = 2;
const F_UPPER: u8 = 3;

// ------------------------------------------------------------- line geometry
struct Geo {
    line_cells: Vec<[usize; 4]>, // 76 lines, flat cell indices
    cell_lines: Vec<Vec<u16>>,   // per cell (0..63) -> line indices through it
}

fn build_geo() -> Geo {
    // Mirror alphazero/connect4_env.py::_gen_lines EXACTLY (same append order).
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
    // BTreeSet (not HashSet): no dependency on std's RNG-seeded hasher (wasm).
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
    let mut cell_lines: Vec<Vec<u16>> = vec![Vec::new(); 64];
    for (li, cells) in line_cells.iter().enumerate() {
        for &c in cells.iter() {
            cell_lines[c].push(li as u16);
        }
    }
    Geo { line_cells, cell_lines }
}

fn geo() -> &'static Geo {
    use std::sync::OnceLock;
    static G: OnceLock<Geo> = OnceLock::new();
    G.get_or_init(build_geo)
}

// -------------------------------------------------------- transposition table
// Packed 16-byte entries in 4-way, 64-byte (cache-line) buckets. Each entry is
// two u64 atomics storing (key ^ data, data) — the XOR trick self-invalidates
// torn writes, so the table is lock-free and safely shareable across threads.
//
// data layout (u64):
//   bits  0..32  val (i32 as u32)          bits 40..42  flag (0 = empty)
//   bits 32..40  depth (u8)                bits 42..48  best col + 1
//   bits 48..56  age (u8)
const TT_WAYS: usize = 4;

pub struct TT {
    slots: Vec<[AtomicU64; 2]>,
    bucket_mask: u64,
    age: AtomicU8,
}

impl TT {
    /// Table with 2^log2_entries entries (16 bytes each).
    pub fn new(log2_entries: usize) -> Self {
        let n = 1usize << log2_entries;
        let buckets = (n / TT_WAYS).max(1);
        let mut slots = Vec::with_capacity(n);
        for _ in 0..n {
            slots.push([AtomicU64::new(0), AtomicU64::new(0)]);
        }
        TT { slots, bucket_mask: (buckets as u64) - 1, age: AtomicU8::new(0) }
    }

    pub fn new_age(&self) {
        self.age.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    fn bucket(&self, hash: u64) -> usize {
        ((hash & self.bucket_mask) as usize) * TT_WAYS
    }

    /// probe -> (val, depth, flag, best_col); best_col = -1 if none stored.
    #[inline]
    fn probe(&self, hash: u64) -> Option<(i32, i32, u8, i32)> {
        let b = self.bucket(hash);
        for i in 0..TT_WAYS {
            let k = self.slots[b + i][0].load(Ordering::Relaxed);
            let d = self.slots[b + i][1].load(Ordering::Relaxed);
            if d != 0 && (k ^ d) == hash {
                let val = d as u32 as i32;
                let depth = ((d >> 32) & 0xFF) as i32;
                let flag = ((d >> 40) & 0x3) as u8;
                let best_raw = ((d >> 42) & 0x3F) as i32;
                let best = if best_raw >= 1 && best_raw <= 16 { best_raw - 1 } else { -1 };
                return Some((val, depth, flag, best));
            }
        }
        None
    }

    #[inline]
    fn store(&self, hash: u64, val: i32, depth: i32, flag: u8, best: i32) {
        let age = self.age.load(Ordering::Relaxed);
        let b = self.bucket(hash);
        let mut victim = 0usize;
        let mut victim_score = i64::MAX; // lower = more replaceable
        for i in 0..TT_WAYS {
            let k = self.slots[b + i][0].load(Ordering::Relaxed);
            let d = self.slots[b + i][1].load(Ordering::Relaxed);
            if d == 0 || (k ^ d) == hash {
                victim = i; // empty slot or same position: take it
                victim_score = i64::MIN;
                break;
            }
            // replacement: evict stale-age first, then shallow (depth-preferred
            // within the same age)
            let e_age = ((d >> 48) & 0xFF) as u8;
            let e_depth = ((d >> 32) & 0xFF) as i64;
            let s = e_depth - (age.wrapping_sub(e_age) as i64) * 64;
            if s < victim_score {
                victim_score = s;
                victim = i;
            }
        }
        let best_bits = if (0..16).contains(&best) { (best + 1) as u64 } else { 0x3F };
        let data = (val as u32 as u64)
            | ((depth.clamp(0, 255) as u64) << 32)
            | ((flag as u64) << 40)
            | (best_bits << 42)
            | ((age as u64) << 48);
        self.slots[b + victim][0].store(hash ^ data, Ordering::Relaxed);
        self.slots[b + victim][1].store(data, Ordering::Relaxed);
    }
}

// --------------------------------------------------------------------- engine
pub struct C4 {
    bb: [u64; 2],
    h: [u8; 16],
    cnt: [[u8; NLINES]; 2],
    score: i32,       // player +1 frame, integer (SCALE = 1.0)
    turn: usize,      // 0 -> +1 to move, 1 -> -1
    wp: [i32; 5],     // [0, W1, W2, W3, 0] scaled
    droppable: u64,   // one bit per playable drop cell
    threat: [u64; 2], // cells completing a line for side
    tcnt: [[u8; 64]; 2],
    nodes: u64,
    stop_fn: Box<dyn Fn(u64) -> bool>,
    stop: bool,
    root_best: i32,
    exact: bool, // true = no NMP/LMR (deterministic exact alpha-beta)
    tt: Arc<TT>,
    killers: [[i32; 2]; MAXPLY],
    history: [u64; 16],
    geo: &'static Geo,
}

impl C4 {
    pub fn new(w1: f64, w2: f64) -> Self {
        Self::with_tt(w1, w2, Arc::new(TT::new(21))) // 2^21 x 16B = 32 MB
    }

    pub fn with_tt(w1: f64, w2: f64, tt: Arc<TT>) -> Self {
        let s = SCALE as f64;
        C4 {
            bb: [0, 0],
            h: [0; 16],
            cnt: [[0; NLINES]; 2],
            score: 0,
            turn: 0,
            wp: [0, (w1 * s).round() as i32, (w2 * s).round() as i32, SCALE, 0],
            droppable: 0,
            threat: [0, 0],
            tcnt: [[0; 64]; 2],
            nodes: 0,
            stop_fn: Box::new(|_| false),
            stop: false,
            root_best: -1,
            exact: false,
            tt,
            killers: [[-1; 2]; MAXPLY],
            history: [0; 16],
            geo: geo(),
        }
    }

    pub fn from_board(board: &[i8], to_play: i8, w1: f64, w2: f64) -> Self {
        let mut e = C4::new(w1, w2);
        e.set_position(board, to_play);
        e
    }

    pub fn set_stop_fn(&mut self, f: Box<dyn Fn(u64) -> bool>) {
        self.stop_fn = f;
    }

    /// Load position from a flat length-64 board (`cell = x*16+y*4+z`),
    /// values in {0,+1,-1}, and side to move (+1/-1).
    pub fn set_position(&mut self, board: &[i8], to_play: i8) {
        self.bb = [0, 0];
        self.h = [0; 16];
        self.cnt = [[0; NLINES]; 2];
        self.threat = [0, 0];
        self.tcnt = [[0; 64]; 2];
        self.droppable = 0;
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
            if filled < 4 {
                self.droppable |= 1u64 << (base + filled as usize);
            }
        }
        let mut score: i32 = 0;
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
                if a == 3 {
                    for &c in cells.iter() {
                        self.tcnt[0][c] += 1;
                        self.threat[0] |= 1u64 << c;
                    }
                }
            } else if b > 0 && a == 0 {
                score -= self.wp[b as usize];
                if b == 3 {
                    for &c in cells.iter() {
                        self.tcnt[1][c] += 1;
                        self.threat[1] |= 1u64 << c;
                    }
                }
            }
        }
        self.score = score;
        self.turn = if to_play == 1 { 0 } else { 1 };
    }

    // -------------------------------------------------------------- make/undo
    #[inline]
    fn threat_add(&mut self, side: usize, li: usize) {
        for &c in self.geo.line_cells[li].iter() {
            let t = &mut self.tcnt[side][c];
            *t += 1;
            if *t == 1 {
                self.threat[side] |= 1u64 << c;
            }
        }
    }

    #[inline]
    fn threat_del(&mut self, side: usize, li: usize) {
        for &c in self.geo.line_cells[li].iter() {
            let t = &mut self.tcnt[side][c];
            *t -= 1;
            if *t == 0 {
                self.threat[side] &= !(1u64 << c);
            }
        }
    }

    #[inline]
    fn make(&mut self, col: usize) {
        let z = self.h[col] as usize;
        let cell = col * 4 + z;
        let s = self.turn;
        let o = s ^ 1;
        self.bb[s] |= 1u64 << cell;
        let sgn: i32 = if s == 0 { 1 } else { -1 };
        let nl = self.geo.cell_lines[cell].len();
        for k in 0..nl {
            let li = self.geo.cell_lines[cell][k] as usize;
            let a = self.cnt[s][li];
            let b = self.cnt[o][li];
            if b == 0 {
                // still pure-ours after the drop
                self.score += sgn * (self.wp[(a + 1) as usize] - self.wp[a as usize]);
                if a + 1 == 3 {
                    self.threat_add(s, li);
                }
            } else if a == 0 {
                // was pure-theirs; becomes mixed
                self.score += sgn * self.wp[b as usize];
                if b == 3 {
                    self.threat_del(o, li);
                }
            }
            self.cnt[s][li] = a + 1;
        }
        self.droppable &= !(1u64 << cell);
        if z + 1 < 4 {
            self.droppable |= 1u64 << (cell + 1);
        }
        self.h[col] = (z + 1) as u8;
        self.turn = o;
    }

    #[inline]
    fn undo(&mut self, col: usize) {
        let o = self.turn;
        let s = o ^ 1;
        self.turn = s;
        let z = (self.h[col] - 1) as usize;
        self.h[col] = z as u8;
        let cell = col * 4 + z;
        self.bb[s] &= !(1u64 << cell);
        if z + 1 < 4 {
            self.droppable &= !(1u64 << (cell + 1));
        }
        self.droppable |= 1u64 << cell;
        let sgn: i32 = if s == 0 { 1 } else { -1 };
        let nl = self.geo.cell_lines[cell].len();
        for k in 0..nl {
            let li = self.geo.cell_lines[cell][k] as usize;
            let a = self.cnt[s][li] - 1;
            let b = self.cnt[o][li];
            self.cnt[s][li] = a;
            if b == 0 {
                self.score -= sgn * (self.wp[(a + 1) as usize] - self.wp[a as usize]);
                if a + 1 == 3 {
                    self.threat_del(s, li);
                }
            } else if a == 0 {
                self.score -= sgn * self.wp[b as usize];
                if b == 3 {
                    self.threat_add(o, li);
                }
            }
        }
    }

    /// Move-ordering delta for dropping in `col`, in the side-to-move frame,
    /// WITHOUT touching the board: parent_score_stm + delta == child heuristic
    /// exactly, and parent score is constant across children, so sorting by
    /// delta reproduces the old make/undo child ordering at ~1/3 the cost.
    #[inline]
    fn order_delta(&self, col: usize) -> i32 {
        let cell = col * 4 + self.h[col] as usize;
        let s = self.turn;
        let o = s ^ 1;
        let mut d: i32 = 0;
        for &li16 in self.geo.cell_lines[cell].iter() {
            let li = li16 as usize;
            let a = self.cnt[s][li];
            let b = self.cnt[o][li];
            if b == 0 {
                d += self.wp[(a + 1) as usize] - self.wp[a as usize];
            } else if a == 0 {
                d += self.wp[b as usize];
            }
        }
        d
    }

    #[inline]
    fn hash(&self) -> u64 {
        // 128 -> 64 mix + splitmix64 finalizer. Side-to-move IS included: with
        // null moves the same (bb0, bb1) occurs with either side to move.
        let mut x = self.bb[0]
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(self.bb[1].rotate_left(32).wrapping_mul(0xC2B2_AE3D_27D4_EB4F))
            .wrapping_add(self.turn as u64);
        x ^= x >> 30;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^ (x >> 31)
    }

    // ---------------------------------------------------------------- search
    fn negamax(&mut self, depth: i32, mut alpha: i32, mut beta: i32, ply: usize,
               null_ok: bool) -> i32 {
        self.nodes += 1;
        if (self.nodes & 2047) == 0 && (self.stop_fn)(self.nodes) {
            self.stop = true;
        }
        if self.stop {
            return 0;
        }

        // mate-distance pruning
        let mating = MATE - ply as i32;
        if mating < beta {
            beta = mating;
            if alpha >= beta {
                return mating;
            }
        }
        let mated = -(MATE - ply as i32);
        if mated > alpha {
            alpha = mated;
            if alpha >= beta {
                return mated;
            }
        }

        if self.droppable == 0 {
            return 0; // full board -> draw
        }

        let side = self.turn;
        let opp = side ^ 1;

        // immediate win: one AND of the threat and droppable bitboards
        let winning = self.threat[side] & self.droppable;
        if winning != 0 {
            if ply == 0 {
                self.root_best = (winning.trailing_zeros() >> 2) as i32;
            }
            return MATE - ply as i32;
        }

        if depth <= 0 {
            return if side == 0 { self.score } else { -self.score };
        }

        // TT probe
        let hash = self.hash();
        let mut tt_move: i32 = -1;
        if let Some((tval, tdepth, tflag, tbest)) = self.tt.probe(hash) {
            tt_move = tbest;
            if tdepth >= depth {
                let mut v = tval;
                if v > MATE_TH {
                    v -= ply as i32;
                } else if v < -MATE_TH {
                    v += ply as i32;
                }
                match tflag {
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

        let under_threat = self.threat[opp] & self.droppable != 0;
        let is_pv = beta - alpha > 1;

        // null-move pruning: hand the opponent a free move; still >= beta =>
        // fail-high. Skipped at PV nodes, in exact mode, when under immediate
        // threat, after a null, and in mate windows.
        if null_ok && !is_pv && !self.exact
            && depth >= 3 && !under_threat && beta < MATE_TH && beta > -MATE_TH
        {
            let r = if depth >= 6 { 3 } else { 2 };
            self.turn = opp;
            let v = -self.negamax(depth - 1 - r, -beta, -beta + 1, ply + 1, false);
            self.turn = side;
            if self.stop {
                return 0;
            }
            if v >= beta {
                return beta;
            }
        }

        // movegen + ordering: TT move, killers, then integer delta + history
        let k0 = if ply < MAXPLY { self.killers[ply][0] } else { -1 };
        let k1 = if ply < MAXPLY { self.killers[ply][1] } else { -1 };
        let mut moves: [usize; 16] = [0; 16];
        let mut keys: [i64; 16] = [0; 16];
        let mut nm = 0usize;
        for col in 0..16usize {
            if self.h[col] < 4 {
                let key: i64 = if col as i32 == tt_move {
                    i64::MAX
                } else {
                    let mut kk = (self.order_delta(col) as i64) * 2_000_000
                        + (self.history[col].min(1_999_999) as i64);
                    if col as i32 == k0 {
                        kk += 4_000_000_000_000;
                    } else if col as i32 == k1 {
                        kk += 2_000_000_000_000;
                    }
                    kk
                };
                let mut j = nm;
                while j > 0 && keys[j - 1] < key {
                    keys[j] = keys[j - 1];
                    moves[j] = moves[j - 1];
                    j -= 1;
                }
                keys[j] = key;
                moves[j] = col;
                nm += 1;
            }
        }

        let a_orig = alpha;
        let mut best = -INF;
        let mut best_col = moves[0];
        for i in 0..nm {
            let col = moves[i];
            self.make(col);
            let val = if i == 0 {
                -self.negamax(depth - 1, -beta, -alpha, ply + 1, true)
            } else {
                // late-move reduction: late, quiet (non-tt/killer), deep enough,
                // not under threat, never in exact mode; gentler at PV nodes.
                let mut r = 0;
                if !self.exact && depth >= 3 && !under_threat
                    && col as i32 != k0 && col as i32 != k1
                    && i >= if is_pv { 5 } else { 3 }
                {
                    r = 1 + ((!is_pv && depth >= 6 && i >= 8) as i32);
                }
                let mut v = -self.negamax(depth - 1 - r, -alpha - 1, -alpha, ply + 1, true);
                if r > 0 && v > alpha && !self.stop {
                    v = -self.negamax(depth - 1, -alpha - 1, -alpha, ply + 1, true);
                }
                if v > alpha && v < beta && !self.stop {
                    v = -self.negamax(depth - 1, -beta, -alpha, ply + 1, true);
                }
                v
            };
            self.undo(col);
            if self.stop {
                return 0;
            }
            if val > best {
                best = val;
                best_col = col;
                if val > alpha {
                    alpha = val;
                }
            }
            if alpha >= beta {
                if ply < MAXPLY && self.killers[ply][0] != col as i32 {
                    self.killers[ply][1] = self.killers[ply][0];
                    self.killers[ply][0] = col as i32;
                }
                self.history[col] += (depth as u64) * (depth as u64);
                break;
            }
        }

        if ply == 0 {
            self.root_best = best_col as i32;
        }

        let mut store_v = best;
        if store_v > MATE_TH {
            store_v += ply as i32;
        } else if store_v < -MATE_TH {
            store_v -= ply as i32;
        }
        let flag = if best <= a_orig {
            F_UPPER
        } else if best >= beta {
            F_LOWER
        } else {
            F_EXACT
        };
        self.tt.store(hash, store_v, depth, flag, best_col as i32);
        best
    }

    fn fallback(&self) -> i32 {
        let win = self.threat[self.turn] & self.droppable;
        if win != 0 {
            return (win.trailing_zeros() >> 2) as i32;
        }
        let block = self.threat[self.turn ^ 1] & self.droppable;
        if block != 0 {
            return (block.trailing_zeros() >> 2) as i32;
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

    fn reset_search_state(&mut self) {
        self.nodes = 0;
        self.stop = false;
        self.root_best = -1;
        self.killers = [[-1; 2]; MAXPLY];
        self.history = [0; 16];
    }

    /// Fixed-depth search (ignores the stop predicate) in EXACT mode: no
    /// null-move pruning, no LMR — deterministic full-strength alpha-beta to
    /// exactly `depth`. This is the verification/tuning path; the timed
    /// production path (`search_id`) keeps all speculative pruning.
    /// Returns (col, depth, nodes, score_i32).
    pub fn search_depth(&mut self, depth: i32) -> (i32, i32, u64, i32) {
        self.reset_search_state();
        self.tt.new_age();
        self.exact = true;
        self.stop_fn = Box::new(|_| false);
        let legal: Vec<usize> = (0..16).filter(|&c| self.h[c] < 4).collect();
        if legal.is_empty() {
            return (-1, 0, 0, 0);
        }
        if legal.len() == 1 {
            return (legal[0] as i32, 0, 0, 0);
        }
        let score = self.negamax(depth, -INF, INF, 0, false);
        let col = if self.root_best >= 0 { self.root_best } else { self.fallback() };
        (col, depth, self.nodes, score)
    }

    /// Iterative deepening until `stop_fn` fires (or solved / exhausted).
    /// Returns (col, depth_completed, nodes, score_i32).
    pub fn search_id(&mut self) -> (i32, i32, u64, i32) {
        self.tt.new_age();
        self.search_id_inner(1)
    }

    /// `search_id` starting at `start_depth` — used by lazy-SMP helpers (which
    /// share the TT and must NOT bump its age; only the main searcher does).
    pub fn search_id_from(&mut self, start_depth: i32) -> (i32, i32, u64, i32) {
        self.search_id_inner(start_depth)
    }

    fn search_id_inner(&mut self, start_depth: i32) -> (i32, i32, u64, i32) {
        self.reset_search_state();
        self.exact = false;
        let legal: Vec<usize> = (0..16).filter(|&c| self.h[c] < 4).collect();
        if legal.is_empty() {
            return (-1, 0, 0, 0);
        }
        if legal.len() == 1 {
            return (legal[0] as i32, 0, 0, 0);
        }

        let mut best = self.fallback();
        let mut best_score: i32 = 0;
        let mut depth_done = 0;
        let empties: i32 = (0..16).map(|c| 4 - self.h[c] as i32).sum();
        let max_depth = empties.min(64);

        let mut prev: i32 = 0;
        let mut depth = start_depth.clamp(1, max_depth);
        while depth <= max_depth {
            self.root_best = -1;
            // integer aspiration window (from depth 5, once prev is meaningful)
            let mut window: i32 = SCALE / 2;
            let (mut alpha, mut beta) = if depth >= 5 {
                (prev.saturating_sub(window), prev.saturating_add(window))
            } else {
                (-INF, INF)
            };
            let mut score;
            loop {
                score = self.negamax(depth, alpha, beta, 0, false);
                if self.stop {
                    break;
                }
                if score <= alpha {
                    window = window.saturating_mul(4);
                    alpha = if window > 100_000_000 { -INF } else { score - window };
                } else if score >= beta {
                    window = window.saturating_mul(4);
                    beta = if window > 100_000_000 { INF } else { score + window };
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

    /// Depth-0 static eval in the side-to-move frame, integer (SCALE = 1.0).
    pub fn static_eval(&self) -> i32 {
        if self.turn == 0 {
            self.score
        } else {
            -self.score
        }
    }

    pub fn empties(&self) -> i32 {
        (0..16).map(|c| 4 - self.h[c] as i32).sum()
    }
}

/// Convert an internal integer score to the external float scale (1.0 = W3).
/// Mate scores map to +/-(100_000 - plies-to-mate).
pub fn score_to_f64(s: i32) -> f64 {
    if s > MATE_TH {
        100_000.0 - (MATE - s) as f64
    } else if s < -MATE_TH {
        -100_000.0 + (MATE + s) as f64
    } else {
        s as f64 / SCALE as f64
    }
}

// ------------------------------------------------------------------ lazy SMP
/// Lazy-SMP search: `threads` independent searchers over one shared lock-free
/// TT; odd helpers start one ply deeper so several horizons fill the table at
/// once. `deadline_check(nodes)` is each thread's stop test (wall clock etc.);
/// any thread tripping it — or the main thread finishing (solved/exhausted) —
/// stops everyone via a shared flag. Native only (wasm: use `search_id`).
/// Returns (col, depth, total_nodes, score_i32) from the deepest finisher.
pub fn search_smp(
    board: &[i8],
    to_play: i8,
    w1: f64,
    w2: f64,
    threads: usize,
    log2_tt: usize,
    deadline_check: impl Fn(u64) -> bool + Send + Sync + 'static,
) -> (i32, i32, u64, i32) {
    let tt = Arc::new(TT::new(log2_tt));
    tt.new_age();
    let stop = Arc::new(AtomicBool::new(false));
    let deadline_check = Arc::new(deadline_check);
    let n = threads.max(1);

    let results: Vec<(i32, i32, u64, i32)> = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for tid in 0..n {
            let tt = Arc::clone(&tt);
            let stop = Arc::clone(&stop);
            let dl = Arc::clone(&deadline_check);
            let board: Vec<i8> = board.to_vec();
            handles.push(scope.spawn(move || {
                let mut eng = C4::with_tt(w1, w2, tt);
                eng.set_position(&board, to_play);
                let stop2 = Arc::clone(&stop);
                eng.set_stop_fn(Box::new(move |nodes| {
                    if stop2.load(Ordering::Relaxed) {
                        return true;
                    }
                    if dl(nodes) {
                        stop2.store(true, Ordering::Relaxed);
                        return true;
                    }
                    false
                }));
                let r = eng.search_id_from(if tid % 2 == 1 { 2 } else { 1 });
                if tid == 0 {
                    stop.store(true, Ordering::Relaxed); // main done -> stop all
                }
                r
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // pick the deepest completed result (main thread wins ties / mates)
    let mut bi = 0usize;
    for i in 1..results.len() {
        if results[i].0 >= 0
            && results[i].1 > results[bi].1
            && !(results[bi].3.abs() > MATE_TH)
        {
            bi = i;
        }
    }
    let mut r = results[bi];
    r.2 = results.iter().map(|x| x.2).sum(); // total nodes across threads
    r
}
