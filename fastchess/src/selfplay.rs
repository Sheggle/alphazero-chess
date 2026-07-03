//! Batched Gumbel self-play engine (Rust port of `alphazero/chess_batched.py`).
//!
//! Runs `n_games` concurrent game-trees in lockstep. Each round, every active
//! game descends its tree to ONE leaf needing a network value, fills that leaf's
//! (18,8,8) planes into a shared batch buffer, and a SINGLE Python `eval_fn` call
//! evaluates the whole batch. Rust then applies the legal-move mask + softmax per
//! game, expands + backpropagates, and continues. The net forward is the only
//! synchronisation point, so batches are `#active games` wide (512-2048).
//!
//! The search math is a faithful, value-preserving port of `GumbelMCTS` /
//! `_BatchedGumbel`: root expand (`root.n=1, root.w=value`), Gumbel-top-k over
//! `gumbel+logits`, Sequential Halving scoring by `gpref + sigma(Q)`,
//! `sigma(q)=(c_visit+max_visit)*c_scale*q`, completed-Q policy target, PUCT below
//! root, negamax backup. Terminal tests use `Board::is_terminal` (claim_draw=True),
//! matching `gumbel.py` exactly. The value *target* reproduces
//! `chess_train.play_chess_game` (material-anchored capped games).
//!
//! Determinism: each game owns a SplitMix64-seeded RNG keyed only by (seed, idx),
//! so output is independent of scheduling/concurrency. Gumbel noise is iid
//! Gumbel(0,1) — distributionally identical to numpy's, but NOT bit-identical, so
//! exact validation is done with `add_noise=false`.

use crate::Board;
use half::f16;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use rayon::prelude::*;
use rayon::ThreadPool;
use std::sync::OnceLock;

const ACTION_SIZE: usize = 4672;
const BUF: usize = 18 * 8 * 8; // 1152

// ----------------------------------------------------------------------------
// Shared rayon pool for the per-round, per-game tree work. The B games are fully
// independent (game i's tree never touches game j), so descend/expand/backprop/
// mask/softmax across games is embarrassingly parallel. Sized to the available
// parallelism (respects container/cgroup vCPU quotas), capped so we never
// oversubscribe absurdly. Built once; a *local* pool (not the global one) so
// repeated `run_selfplay` calls in the same process never hit build_global's
// "already initialised" error.
// ----------------------------------------------------------------------------
static POOL: OnceLock<ThreadPool> = OnceLock::new();
fn pool() -> &'static ThreadPool {
    POOL.get_or_init(|| {
        let n = std::env::var("FASTCHESS_THREADS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .or_else(|| std::thread::available_parallelism().ok().map(|x| x.get()))
            .unwrap_or(4)
            .clamp(1, 64);
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .expect("failed to build fastchess rayon pool")
    })
}

/// Send/Sync wrapper around a raw read-only pointer into a numpy buffer, so the
/// network result can be scattered to per-game threads with zero copy. The
/// pointed-to arrays (`logits`/`values`) are kept alive on the GIL-holding parent
/// thread for the whole `allow_threads` scope, and every game reads a disjoint,
/// read-only slice, so this is sound.
#[derive(Clone, Copy)]
struct RoPtr(*const f32);
unsafe impl Send for RoPtr {}
unsafe impl Sync for RoPtr {}
impl RoPtr {
    /// SAFETY: caller guarantees `[off, off+len)` lies within the live buffer.
    /// Taking `self` by value makes closures capture the whole (Sync) wrapper
    /// rather than borrowing the inner `*const f32` (which is not Sync).
    #[inline]
    unsafe fn slice(self, off: usize, len: usize) -> &'static [f32] {
        std::slice::from_raw_parts(self.0.add(off), len)
    }
    #[inline]
    unsafe fn at(self, off: usize) -> f32 {
        *self.0.add(off)
    }
}

// ----------------------------------------------------------------------------
// RNG: SplitMix64 -> uniform double -> Gumbel(0,1). Per game, seeded by index.
// ----------------------------------------------------------------------------
struct GameRng {
    state: u64,
}
impl GameRng {
    fn seeded(seed: u64, idx: u64) -> Self {
        // Mix the global seed with the game index so games are independent.
        let mut s = seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(idx.wrapping_mul(0xD1B5_4A32_D192_ED03))
            .wrapping_add(0x1234_5678_9ABC_DEF0);
        // a couple of mixing rounds
        for _ in 0..3 {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            s = z ^ (z >> 31);
        }
        GameRng { state: s }
    }
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    #[inline]
    fn next_f64(&mut self) -> f64 {
        // 53-bit mantissa in [0,1)
        (self.next_u64() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }
    /// Standard Gumbel(0,1): G = -ln(-ln(U)), U ~ Uniform(0,1).
    #[inline]
    fn gumbel(&mut self) -> f64 {
        let mut u = self.next_f64();
        if u <= 0.0 {
            u = f64::MIN_POSITIVE;
        }
        let inner = -u.ln(); // -ln(u) > 0
        let inner = if inner <= 0.0 { f64::MIN_POSITIVE } else { inner };
        -inner.ln()
    }
}

// ----------------------------------------------------------------------------
// Hyper-parameters shared by all games.
// ----------------------------------------------------------------------------
#[derive(Clone, Copy)]
struct Cfg {
    n_sims: usize,
    max_considered: usize,
    c_visit: f64,
    c_scale: f64,
    c_puct: f64,
    add_noise: bool,
}

// ----------------------------------------------------------------------------
// Lazy MCTS node (arena-allocated; children materialised on first selection).
// ----------------------------------------------------------------------------
struct Node {
    to_play: i8,
    n: u32,
    w: f64,
    prior: f64,
    expanded: bool,
    is_terminal: i8, // -1 unknown, 0 false, 1 true
    term_value: f64, // in this node's to-move perspective
    legal_moves: Vec<shakmaty::Move>,
    legal_actions: Vec<i32>,
    priors: Vec<f64>, // aligned to legal_*
    children: Vec<i32>, // node id or -1, aligned to legal_*
}
impl Node {
    fn unexpanded(prior: f64, to_play: i8) -> Self {
        Node {
            to_play,
            n: 0,
            w: 0.0,
            prior,
            expanded: false,
            is_terminal: -1,
            term_value: 0.0,
            legal_moves: Vec::new(),
            legal_actions: Vec::new(),
            priors: Vec::new(),
            children: Vec::new(),
        }
    }
    #[inline]
    fn q(&self) -> f64 {
        if self.n > 0 {
            self.w / self.n as f64
        } else {
            0.0
        }
    }
}

// A paused simulation awaiting a network value for its leaf.
struct Pending {
    path: Vec<usize>, // node ids, child .. leaf
    leaf_moves: Vec<shakmaty::Move>,
    leaf_actions: Vec<i32>,
}

#[derive(PartialEq)]
enum Phase {
    NeedRootEval,
    Running,
    Done,
}

enum SimResult {
    NeedEval,
    Terminal,
}

// ----------------------------------------------------------------------------
// One game's in-flight Gumbel search (a hand-rolled state machine standing in
// for chess_batched's coroutine).
// ----------------------------------------------------------------------------
struct Search {
    cfg: Cfg,
    arena: Vec<Node>,
    root_id: usize,
    root_board: Board,
    root_value: f64,
    logits: Vec<f64>, // log(softmax priors) over root legal
    gumbel: Vec<f64>, // gumbel noise per root legal (0 if !add_noise)
    gpref: Vec<f64>,  // gumbel[j] + logits[j]
    // The ORIGINAL m candidate actions (order[:m]). The final move is chosen over
    // THIS set, not the halving survivors -- gumbel.py's _sequential_halving
    // mutates only a local copy, so a dropped candidate (with its few-sim Q) can
    // still be the final pick. `sh` is the mutable round-robin/halving working set.
    considered: Vec<usize>,
    sh: Vec<usize>,
    // sequential-halving driver state
    budget: usize,
    used: usize,
    num_phases: usize,
    ci: usize,
    pj: usize,
    per: usize,
    phase_active: bool,
    // current outstanding eval request
    cur_planes: Vec<f32>,
    cur_legal: Vec<i32>,
    pending: Option<Pending>,
    phase: Phase,
    // result of a completed search
    result_best_j: usize,
    result_pi_idx: Vec<i32>,
    result_pi_val: Vec<f32>,
}

impl Search {
    fn new(board: Board, rng: &mut GameRng, cfg: Cfg) -> Self {
        let to_play = board.to_play_i8();
        let moves = board.legal_moves_vec();
        let actions: Vec<i32> = moves.iter().map(|m| board.action_index(m)).collect();
        let n_leg = moves.len();
        let mut root = Node::unexpanded(0.0, to_play);
        root.legal_moves = moves;
        root.legal_actions = actions.clone();
        root.children = vec![-1; n_leg];
        let gumbel: Vec<f64> = if cfg.add_noise {
            (0..n_leg).map(|_| rng.gumbel()).collect()
        } else {
            vec![0.0; n_leg]
        };
        let mut cur_planes = vec![0f32; BUF];
        board.encode_buf(&mut cur_planes);
        Search {
            cfg,
            arena: vec![root],
            root_id: 0,
            root_board: board,
            root_value: 0.0,
            logits: Vec::new(),
            gumbel,
            gpref: Vec::new(),
            considered: Vec::new(),
            sh: Vec::new(),
            budget: cfg.n_sims,
            used: 0,
            num_phases: 1,
            ci: 0,
            pj: 0,
            per: 0,
            phase_active: false,
            cur_planes,
            cur_legal: actions,
            pending: None,
            phase: Phase::NeedRootEval,
            result_best_j: 0,
            result_pi_idx: Vec::new(),
            result_pi_val: Vec::new(),
        }
    }

    #[inline]
    fn done(&self) -> bool {
        self.phase == Phase::Done
    }

    // ---- arena helpers ----
    #[inline]
    fn child_id_root(&self, j: usize) -> i32 {
        self.arena[self.root_id].children[j]
    }
    #[inline]
    fn child_n_root(&self, j: usize) -> u32 {
        let c = self.child_id_root(j);
        if c < 0 {
            0
        } else {
            self.arena[c as usize].n
        }
    }
    #[inline]
    fn q_root(&self, j: usize) -> f64 {
        let c = self.child_id_root(j);
        if c >= 0 && self.arena[c as usize].n > 0 {
            -self.arena[c as usize].q()
        } else {
            0.0
        }
    }
    fn max_visit(&self) -> u32 {
        let root = &self.arena[self.root_id];
        let mut m = 0u32;
        for &cid in &root.children {
            if cid >= 0 {
                let n = self.arena[cid as usize].n;
                if n > m {
                    m = n;
                }
            }
        }
        m
    }
    #[inline]
    fn sigma(&self, max_visit: u32, q: f64) -> f64 {
        (self.cfg.c_visit + max_visit as f64) * self.cfg.c_scale * q
    }

    fn ensure_child(&mut self, parent_id: usize, j: usize) -> usize {
        let cid = self.arena[parent_id].children[j];
        if cid >= 0 {
            return cid as usize;
        }
        let to_play = -self.arena[parent_id].to_play;
        let prior = self.arena[parent_id].priors[j];
        let id = self.arena.len();
        self.arena.push(Node::unexpanded(prior, to_play));
        self.arena[parent_id].children[j] = id as i32;
        id
    }

    fn select_child(&self, node_id: usize) -> (usize, shakmaty::Move) {
        let node = &self.arena[node_id];
        let sqrt_n = (node.n as f64).sqrt();
        let cp = self.cfg.c_puct;
        let mut best_j = 0usize;
        let mut best = f64::NEG_INFINITY;
        for j in 0..node.legal_moves.len() {
            let cid = node.children[j];
            let (cn, cq) = if cid < 0 {
                (0u32, 0.0f64)
            } else {
                let c = &self.arena[cid as usize];
                (c.n, c.q())
            };
            let u = cp * node.priors[j] * sqrt_n / (1.0 + cn as f64);
            let score = -cq + u; // strict '>' -> ties keep legal order
            if score > best {
                best = score;
                best_j = j;
            }
        }
        (best_j, node.legal_moves[best_j].clone())
    }

    fn backprop(&mut self, path: &[usize], value: f64) {
        let mut v = value;
        for &nd in path.iter().rev() {
            let n = &mut self.arena[nd];
            n.n += 1;
            n.w += v;
            v = -v;
        }
    }

    /// One simulation forced through root -> considered[j0], PUCT below.
    fn simulate(&mut self, j0: usize) -> SimResult {
        let mut cur = self.root_board.dup();
        let rm = self.arena[self.root_id].legal_moves[j0].clone();
        cur.play_move(&rm);
        let child_id = self.ensure_child(self.root_id, j0);
        let mut path = vec![child_id];
        let mut node_id = child_id;

        loop {
            let (expanded, term) = {
                let n = &self.arena[node_id];
                (n.expanded, n.is_terminal)
            };
            if !expanded || term == 1 {
                break;
            }
            let (bj, mv) = self.select_child(node_id);
            cur.play_move(&mv);
            let cid = self.ensure_child(node_id, bj);
            path.push(cid);
            node_id = cid;
        }

        // Cheap in-search terminal test (claim_draw=True, like gumbel.py).
        if self.arena[node_id].is_terminal == -1 {
            if cur.is_terminal_b() {
                let tp = self.arena[node_id].to_play as f64;
                let v = cur.result_white() as f64 * tp;
                let n = &mut self.arena[node_id];
                n.is_terminal = 1;
                n.term_value = v;
            } else {
                self.arena[node_id].is_terminal = 0;
            }
        }

        if self.arena[node_id].is_terminal == 1 {
            let v = self.arena[node_id].term_value;
            self.backprop(&path, v);
            self.arena[self.root_id].n += 1;
            SimResult::Terminal
        } else {
            // Non-terminal unexpanded leaf -> needs a network value.
            let moves = cur.legal_moves_vec();
            let actions: Vec<i32> = moves.iter().map(|m| cur.action_index(m)).collect();
            cur.encode_buf(&mut self.cur_planes);
            self.cur_legal = actions.clone();
            self.pending = Some(Pending {
                path,
                leaf_moves: moves,
                leaf_actions: actions,
            });
            SimResult::NeedEval
        }
    }

    /// Next root action to simulate under the Sequential-Halving schedule, or
    /// None when the budget/halving is exhausted. Mirrors `_sequential_halving`.
    fn next_action(&mut self) -> Option<usize> {
        loop {
            if !self.phase_active {
                if self.used >= self.budget || self.sh.is_empty() {
                    return None;
                }
                let len = self.sh.len();
                self.per = if len > 1 {
                    std::cmp::max(1, (self.budget / self.num_phases) / len)
                } else {
                    self.budget
                };
                self.ci = 0;
                self.pj = 0;
                self.phase_active = true;
            }
            while self.ci < self.sh.len() {
                if self.pj < self.per {
                    if self.used >= self.budget {
                        self.phase_active = false;
                        break;
                    }
                    let a = self.sh[self.ci];
                    self.pj += 1;
                    self.used += 1;
                    return Some(a);
                } else {
                    self.pj = 0;
                    self.ci += 1;
                }
            }
            // Phase finished (round-robin done or budget hit).
            self.phase_active = false;
            if self.sh.len() <= 1 || self.used >= self.budget {
                return None;
            }
            // Keep the better half by gpref + sigma(Q) (stable, ties keep order).
            let mv = self.max_visit();
            let mut scored: Vec<(usize, f64)> = self
                .sh
                .iter()
                .map(|&j| (j, self.gpref[j] + self.sigma(mv, self.q_root(j))))
                .collect();
            scored.sort_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            let keep = std::cmp::max(1, scored.len() / 2);
            self.sh = scored[..keep].iter().map(|&(j, _)| j).collect();
        }
    }

    /// Resume with the network result for the outstanding request, then advance
    /// the search until the next request or completion.
    fn resume(&mut self, priors: Vec<f64>, value: f64) {
        match self.phase {
            Phase::NeedRootEval => {
                {
                    let root = &mut self.arena[self.root_id];
                    root.priors = priors;
                    root.expanded = true;
                    root.is_terminal = 0;
                    root.n = 1;
                    root.w = value;
                }
                self.root_value = value;
                // logits = ln(clip(priors, 1e-12, 1))
                self.logits = self.arena[self.root_id]
                    .priors
                    .iter()
                    .map(|&p| p.clamp(1e-12, 1.0).ln())
                    .collect();
                let l = self.logits.len();
                self.gpref = (0..l).map(|j| self.gumbel[j] + self.logits[j]).collect();
                // m = min(max_considered, len, max(2, n_sims))
                let m = self
                    .cfg
                    .max_considered
                    .min(l)
                    .min(std::cmp::max(2, self.cfg.n_sims));
                // order = argsort(-(gumbel+logits)) == descending gpref (stable).
                let mut order: Vec<usize> = (0..l).collect();
                order.sort_by(|&a, &b| {
                    self.gpref[b]
                        .partial_cmp(&self.gpref[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                self.considered = order[..m].to_vec();
                self.sh = self.considered.clone();
                self.budget = self.cfg.n_sims;
                self.used = 0;
                self.num_phases = if self.considered.len() > 1 {
                    let len = self.considered.len();
                    // ceil(log2(len)) via integer bit width.
                    std::cmp::max(1, (usize::BITS - (len - 1).leading_zeros()) as usize)
                } else {
                    1
                };
                self.phase_active = false;
                self.phase = Phase::Running;
                self.advance();
            }
            Phase::Running => {
                let pending = self.pending.take().expect("resume without pending");
                let leaf_id = *pending.path.last().unwrap();
                {
                    let n_leg = pending.leaf_moves.len();
                    let leaf = &mut self.arena[leaf_id];
                    leaf.priors = priors;
                    leaf.legal_moves = pending.leaf_moves;
                    leaf.legal_actions = pending.leaf_actions;
                    leaf.children = vec![-1; n_leg];
                    leaf.expanded = true;
                    leaf.is_terminal = 0;
                }
                self.backprop(&pending.path, value);
                self.arena[self.root_id].n += 1;
                self.advance();
            }
            Phase::Done => unreachable!("resume after done"),
        }
    }

    /// Run simulations until one needs a network value (returns, leaving the
    /// request in cur_planes/cur_legal) or the search completes (-> Done).
    fn advance(&mut self) {
        loop {
            match self.next_action() {
                None => {
                    self.finalize();
                    self.phase = Phase::Done;
                    return;
                }
                Some(j) => match self.simulate(j) {
                    SimResult::NeedEval => return,
                    SimResult::Terminal => continue,
                },
            }
        }
    }

    /// Completed-Q policy target + final Gumbel move choice. Mirrors
    /// `_completed_policy` and the `best = max(considered, ...)` selection.
    fn finalize(&mut self) {
        let l = self.logits.len();
        // p2 = softmax(logits) (recovers normalized priors)
        let maxlog = self.logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut p2: Vec<f64> = self.logits.iter().map(|&x| (x - maxlog).exp()).collect();
        let s2: f64 = p2.iter().sum();
        for x in &mut p2 {
            *x /= s2;
        }
        let mut n_total: u64 = 0;
        let mut visited = false;
        let mut sum_p = 0.0f64;
        let mut wq_num = 0.0f64;
        for j in 0..l {
            let cn = self.child_n_root(j);
            n_total += cn as u64;
            if cn > 0 {
                visited = true;
                sum_p += p2[j];
                wq_num += p2[j] * self.q_root(j);
            }
        }
        let v_mix = if visited {
            let wq = wq_num / sum_p.max(1e-12);
            (self.root_value + n_total as f64 * wq) / (1.0 + n_total as f64)
        } else {
            self.root_value
        };
        let mv = self.max_visit();
        let mut score = vec![0.0f64; l];
        for j in 0..l {
            let cq = if self.child_n_root(j) > 0 {
                self.q_root(j)
            } else {
                v_mix
            };
            score[j] = self.logits[j] + self.sigma(mv, cq);
        }
        let ms = score.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut ex: Vec<f64> = score.iter().map(|&s| (s - ms).exp()).collect();
        let se: f64 = ex.iter().sum();
        for x in &mut ex {
            *x /= se;
        }
        // Sparse pi: (action, prob f32), drop f32-zeros, sort by action asc to
        // match numpy np.nonzero ordering.
        let mut pairs: Vec<(i32, f32)> = Vec::with_capacity(l);
        for j in 0..l {
            let v = ex[j] as f32;
            if v != 0.0 {
                pairs.push((self.arena[self.root_id].legal_actions[j], v));
            }
        }
        pairs.sort_by_key(|&(a, _)| a);
        self.result_pi_idx = pairs.iter().map(|&(a, _)| a).collect();
        self.result_pi_val = pairs.iter().map(|&(_, v)| v).collect();

        // Final action: best considered by gpref + sigma(Q) (first max).
        let mut best_j = self.considered[0];
        let mut best = f64::NEG_INFINITY;
        for &j in &self.considered {
            let sc = self.gpref[j] + self.sigma(mv, self.q_root(j));
            if sc > best {
                best = sc;
                best_j = j;
            }
        }
        self.result_best_j = best_j;
    }

    fn chosen_move(&self) -> shakmaty::Move {
        self.arena[self.root_id].legal_moves[self.result_best_j].clone()
    }
    fn chosen_action(&self) -> i32 {
        self.arena[self.root_id].legal_actions[self.result_best_j]
    }
}

// ----------------------------------------------------------------------------
// Per-game driver bookkeeping.
// ----------------------------------------------------------------------------
struct Rec {
    planes: Vec<f32>,
    to_play: i8,
    mat_w: i32,
    pi_idx: Vec<i32>,
    pi_val: Vec<f32>,
}

struct Game {
    board: Board,
    ply: u32,
    rng: GameRng,
    recs: Vec<Rec>,
    moves: Vec<i32>, // chosen action sequence (for validation)
    search: Option<Search>,
    finished: bool,
}

/// Softmax over a game's legal logits, in f32 (mirrors numpy's float32
/// `_make_batch_eval` / `ChessEvaluator.predict` masking), returned as f64.
/// `ll_in` is the contiguous segment of legal logits for one game, already
/// gathered on-GPU in the same per-game legal order Rust uses — so the values
/// (and hence the softmax) are bit-identical to indexing the full logits row.
fn softmax_legal(ll_in: &[f32]) -> Vec<f64> {
    let mut ll: Vec<f32> = ll_in.to_vec();
    let m = ll.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0f32;
    for x in &mut ll {
        *x = (*x - m).exp();
        s += *x;
    }
    ll.iter().map(|&x| (x / s) as f64).collect()
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (eval_fn, n_games, sims, max_considered, c_visit, c_scale, c_puct, max_ply, mat_thresh, add_noise, seed))]
pub fn run_selfplay(
    py: Python<'_>,
    eval_fn: PyObject,
    n_games: usize,
    sims: usize,
    max_considered: usize,
    c_visit: f64,
    c_scale: f64,
    c_puct: f64,
    max_ply: u32,
    mat_thresh: f64,
    add_noise: bool,
    seed: u64,
) -> PyResult<(PyObject, PyObject)> {
    let cfg = Cfg {
        n_sims: sims,
        max_considered,
        c_visit,
        c_scale,
        c_puct,
        add_noise,
    };

    // Spawn all games; start each one's search (-> a root eval request).
    let mut games: Vec<Game> = Vec::with_capacity(n_games);
    for i in 0..n_games {
        let board = Board::start_pos();
        let mut rng = GameRng::seeded(seed, i as u64);
        let playable = !board.is_terminal_b() && 0 < max_ply;
        let search = if playable {
            Some(Search::new(board.dup(), &mut rng, cfg))
        } else {
            None
        };
        games.push(Game {
            board,
            ply: 0,
            rng,
            recs: Vec::new(),
            moves: Vec::new(),
            search,
            finished: !playable,
        });
    }

    loop {
        // Active = games with an outstanding eval request.
        let active: Vec<usize> = (0..n_games)
            .filter(|&i| games[i].search.is_some() && !games[i].finished)
            .collect();
        if active.is_empty() {
            break;
        }
        let b = active.len();

        // row_of[gi] = the batch row for active game gi, or usize::MAX if idle.
        // Lets each game thread find its own eval-result slice by self-index.
        let mut row_of = vec![usize::MAX; n_games];
        for (row, &gi) in active.iter().enumerate() {
            row_of[gi] = row;
        }

        // Build the batch buffer (B,18,8,8) and per-row legal masks in parallel
        // (GIL released): each row is written by exactly one game, reading only
        // that game's immutable search state -> no shared mutable state.
        let mut batch = vec![0f32; b * BUF];
        let mut legals: Vec<Vec<i32>> = vec![Vec::new(); b];
        {
            let games_ref = &games;
            let active_ref = &active;
            py.allow_threads(|| {
                pool().install(|| {
                    batch
                        .par_chunks_mut(BUF)
                        .zip(legals.par_iter_mut())
                        .enumerate()
                        .for_each(|(row, (chunk, leg))| {
                            let s = games_ref[active_ref[row]].search.as_ref().unwrap();
                            chunk.copy_from_slice(&s.cur_planes);
                            *leg = s.cur_legal.clone();
                        });
                });
            });
        }

        // Flat legal (row, col) indices for an on-GPU gather, in the EXACT
        // per-game order Rust will softmax. This lets eval_fn return only the
        // legal logits (~B*35 floats) rather than the full (B,4672) tensor,
        // killing the dominant per-round D2H. offsets[row] = start of game
        // `row`'s segment in the returned flat legal_logits.
        let mut offsets = vec![0usize; b];
        let mut m_total = 0usize;
        for (row, leg) in legals.iter().enumerate() {
            offsets[row] = m_total;
            m_total += leg.len();
        }
        let mut legal_rows = vec![0i64; m_total];
        let mut legal_cols = vec![0i64; m_total];
        for (row, leg) in legals.iter().enumerate() {
            let base = offsets[row];
            for (j, &a) in leg.iter().enumerate() {
                legal_rows[base + j] = row as i64;
                legal_cols[base + j] = a as i64;
            }
        }

        // ONE batched network forward (the only sync point). torch releases the
        // GIL internally during the CUDA forward, gathers logits[rows, cols] on
        // GPU, and D2Hs only the flat legal logits + per-game values.
        let planes_arr = PyArray1::from_vec_bound(py, batch).reshape([b, 18, 8, 8])?;
        let rows_arr = PyArray1::from_vec_bound(py, legal_rows);
        let cols_arr = PyArray1::from_vec_bound(py, legal_cols);
        let res = eval_fn.call1(py, (planes_arr, rows_arr, cols_arr))?;
        let bound = res.bind(py);
        let tup = bound.downcast::<PyTuple>()?;
        let legal_logits: PyReadonlyArray1<f32> = tup.get_item(0)?.extract()?;
        let values: PyReadonlyArray1<f32> = tup.get_item(1)?.extract()?;
        let ll_s = legal_logits.as_slice()?;
        let values_s = values.as_slice()?;

        // Resume every active game with its masked priors + value IN PARALLEL,
        // with the GIL released. Each game touches only its own arena / RNG /
        // recs; the only cross-thread data are the read-only network results,
        // scattered per-game by its contiguous legal-logit segment. `RoPtr`
        // carries the numpy buffers' base pointers across threads;
        // `legal_logits`/`values` stay alive on this thread for the whole scope,
        // and every game's segment is disjoint + read-only.
        let ll_ptr = RoPtr(ll_s.as_ptr());
        let values_ptr = RoPtr(values_s.as_ptr());
        {
            let legals_ref = &legals;
            let offsets_ref = &offsets;
            let row_of_ref = &row_of;
            py.allow_threads(|| {
                pool().install(|| {
                    games.par_iter_mut().enumerate().for_each(|(gi, g)| {
                        let row = row_of_ref[gi];
                        if row == usize::MAX {
                            return; // idle this round
                        }
                        let len = legals_ref[row].len();
                        // SAFETY: disjoint, read-only segment of a live numpy buffer.
                        let seg = unsafe { ll_ptr.slice(offsets_ref[row], len) };
                        let value = unsafe { values_ptr.at(row) } as f64;
                        let priors = softmax_legal(seg);
                        let s = g.search.as_mut().unwrap();
                        s.resume(priors, value);
                        // If the search completed, record + apply the move and
                        // start the next search, so every active game ends the
                        // round holding a fresh eval request.
                        if s.done() {
                            handle_move(g, cfg, max_ply);
                        }
                    });
                });
            });
        }
    }

    // Finalize: build samples + stats.
    let samples = PyList::empty_bound(py);
    let stats = PyList::empty_bound(py);
    for g in &games {
        let terminal = g.board.is_terminal_b();
        let result_white = g.board.result_white();
        let z_white = if terminal {
            result_white
        } else {
            let d = g.board.material_white() as f64;
            if d >= mat_thresh {
                1
            } else if d <= -mat_thresh {
                -1
            } else {
                0
            }
        };

        for rec in &g.recs {
            // "Just who wins": binary game outcome (terminal result, or the
            // material-adjudicated winner for capped games — see z_white above)
            // from this position's side-to-move perspective. No graded-material
            // term, so a material sacrifice that leads to a won game scores +1.
            let z: f32 = (z_white * rec.to_play as i32) as f32;
            let planes16: Vec<f16> = rec.planes.iter().map(|&x| f16::from_f32(x)).collect();
            let planes_arr = PyArray1::from_vec_bound(py, planes16).reshape([18, 8, 8])?;
            let idx_arr = PyArray1::from_vec_bound(py, rec.pi_idx.iter().map(|&x| x as i16).collect::<Vec<i16>>());
            let val_arr = PyArray1::from_vec_bound(py, rec.pi_val.clone());
            let tup = PyTuple::new_bound(
                py,
                &[
                    planes_arr.into_any().unbind(),
                    idx_arr.into_any().unbind(),
                    val_arr.into_any().unbind(),
                    z.into_pyobject_or_py(py),
                ],
            );
            samples.append(tup)?;
        }

        let d = PyDict::new_bound(py);
        d.set_item("terminal", terminal)?;
        d.set_item("plies", g.ply)?;
        d.set_item("z_white", z_white)?;
        if terminal {
            d.set_item("result", result_white)?;
        } else {
            d.set_item("result", py.None())?;
        }
        d.set_item("moves", g.moves.clone())?;
        stats.append(d)?;
    }

    Ok((samples.into_any().unbind(), stats.into_any().unbind()))
}

/// Record the just-decided move's sample, apply it, and (if the game continues)
/// start the next search; otherwise mark the game finished.
fn handle_move(g: &mut Game, cfg: Cfg, max_ply: u32) {
    let search = g.search.take().unwrap();
    let mv = search.chosen_move();
    let action = search.chosen_action();
    let pi_idx = search.result_pi_idx.clone();
    let pi_val = search.result_pi_val.clone();

    // Sample is recorded at the position BEFORE the move (the search root).
    let mut planes = vec![0f32; BUF];
    g.board.encode_buf(&mut planes);
    let to_play = g.board.to_play_i8();
    let mat_w = g.board.material_white();
    g.recs.push(Rec {
        planes,
        to_play,
        mat_w,
        pi_idx,
        pi_val,
    });
    g.moves.push(action);

    g.board.play_move(&mv);
    g.ply += 1;

    if !g.board.is_terminal_b() && g.ply < max_ply {
        g.search = Some(Search::new(g.board.dup(), &mut g.rng, cfg));
    } else {
        g.finished = true;
    }
}

// Helper: small shim so f32 -> PyObject works across pyo3 0.22 without pulling
// in IntoPyObject churn.
trait IntoPyObjectOrPy {
    fn into_pyobject_or_py(self, py: Python<'_>) -> PyObject;
}
impl IntoPyObjectOrPy for f32 {
    fn into_pyobject_or_py(self, py: Python<'_>) -> PyObject {
        self.into_py(py)
    }
}
