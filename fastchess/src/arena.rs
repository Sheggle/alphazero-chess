//! Leaf-parallel PUCT arena, entirely in Rust. Python is called for exactly one
//! thing per round: the batched (compiled) GPU forward (`eval_fn`). This is the
//! `run_selfplay` pattern (Rust tree + Python eval callback) extended to
//! leaf parallelism + two nets (one per side to move).
//!
//! Per game, per move: leaf-parallel PUCT. Each round runs `L` PUCT descents from
//! the root; a virtual loss is applied along each descended path so the `L`
//! descents diverge (in this node convention `w` is the node's own to-move value,
//! so a virtual *loss for the mover above* is `n+=1, w+=1` on the descended
//! children -> their `q` rises -> `-q` falls -> discouraged). The collected leaf
//! encodings go to one batched `eval_fn` call -> (logits, values); then the
//! virtual loss is removed and the real expand + negamax backup is applied.
//! Terminal leaves are backed up immediately (no eval).
//!
//! Time budget: rounds run until `ms_per_move` wall-clock elapses for that move,
//! then the argmax-visit move is played. `eval_fn_a`/`eval_fn_b` alternate by side
//! to move. Colors are balanced across halves. Games end at terminal or
//! `max_ply`, with material adjudication of caps.
//!
//! L=1 (virtual loss applied then removed with no intervening pass) reproduces
//! ordinary sequential PUCT bit-for-bit.

use crate::Board;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::time::Instant;

const ACTION_SIZE: usize = 4672;
const BUF: usize = 18 * 8 * 8;

// ---------------------------------------------------------------------------
// Per-game deterministic RNG (SplitMix64), used only for opening sampling.
// ---------------------------------------------------------------------------
struct GameRng {
    state: u64,
}
impl GameRng {
    fn seeded(seed: u64, idx: u64) -> Self {
        let mut s = seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(idx.wrapping_mul(0xD1B5_4A32_D192_ED03))
            .wrapping_add(0x1234_5678_9ABC_DEF0);
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
        (self.next_u64() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }
}

// ---------------------------------------------------------------------------
struct Node {
    to_play: i8,
    n: u32,
    w: f64,
    prior: f64,
    expanded: bool,
    is_terminal: i8, // -1 unknown, 0 false, 1 true
    term_value: f64,
    legal_moves: Vec<shakmaty::Move>,
    legal_actions: Vec<i32>,
    priors: Vec<f64>,
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
}

// A leaf awaiting a network value within the current round.
struct Pending {
    path: Vec<usize>,
    leaf_id: usize,
    row: usize, // row in the round's shared batch
    is_root: bool,
    leaf_moves: Vec<shakmaty::Move>,
    leaf_actions: Vec<i32>,
}

#[derive(PartialEq, Clone, Copy)]
enum Phase {
    NeedRoot,
    Running,
}

/// One game: a board the match advances, plus the in-flight search tree for the
/// current move.
struct Game {
    board: Board,
    ply: u32,
    rng: GameRng,
    arena: Vec<Node>,
    root_id: usize,
    root_board: Board,
    phase: Phase,
    pending: Vec<Pending>,
    sims: u64,
    finished: bool,
    moves: Vec<i32>,
}

#[inline]
fn softmax_gather(logits_row: &[f32], legal: &[i32]) -> Vec<f64> {
    let mut ll: Vec<f32> = legal.iter().map(|&a| logits_row[a as usize]).collect();
    let m = ll.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0f32;
    for x in &mut ll {
        *x = (*x - m).exp();
        s += *x;
    }
    ll.iter().map(|&x| (x / s) as f64).collect()
}

impl Game {
    fn new(board: Board, rng: GameRng) -> Self {
        let rb = board.dup();
        Game {
            board,
            ply: 0,
            rng,
            arena: Vec::new(),
            root_id: 0,
            root_board: rb,
            phase: Phase::NeedRoot,
            pending: Vec::new(),
            sims: 0,
            finished: false,
            moves: Vec::new(),
        }
    }

    /// Begin a fresh search for the current board (a new move).
    fn start_search(&mut self) {
        self.root_board = self.board.dup();
        let to_play = self.root_board.to_play_i8();
        self.arena = vec![Node::unexpanded(0.0, to_play)];
        self.root_id = 0;
        self.phase = Phase::NeedRoot;
        self.pending.clear();
        self.sims = 0;
    }

    #[inline]
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

    fn select_child(&self, node_id: usize, c_puct: f64) -> (usize, shakmaty::Move) {
        let node = &self.arena[node_id];
        let sqrt_n = (node.n as f64).sqrt();
        let mut best_j = 0usize;
        let mut best = f64::NEG_INFINITY;
        for j in 0..node.legal_moves.len() {
            let cid = node.children[j];
            let (cn, cq) = if cid < 0 {
                (0u32, 0.0f64)
            } else {
                let c = &self.arena[cid as usize];
                (c.n, if c.n > 0 { c.w / c.n as f64 } else { 0.0 })
            };
            let u = c_puct * node.priors[j] * sqrt_n / (1.0 + cn as f64);
            let score = -cq + u; // strict '>' -> ties keep legal order
            if score > best {
                best = score;
                best_j = j;
            }
        }
        (best_j, node.legal_moves[best_j].clone())
    }

    fn backup(&mut self, path: &[usize], value: f64) {
        let mut v = value;
        for &nd in path.iter().rev() {
            let n = &mut self.arena[nd];
            n.n += 1;
            n.w += v;
            v = -v;
        }
    }

    /// One PUCT descent: pushes a leaf into `batch` (and `self.pending`) if it
    /// needs a network value, or backs up immediately if terminal.
    fn descend_one(&mut self, c_puct: f64, batch: &mut Vec<f32>) {
        let mut cur = self.root_board.dup();
        let mut node_id = self.root_id;
        let mut path = vec![self.root_id];
        loop {
            let (expanded, term) = {
                let nd = &self.arena[node_id];
                (nd.expanded, nd.is_terminal)
            };
            if !expanded || term == 1 {
                break;
            }
            let (bj, mv) = self.select_child(node_id, c_puct);
            cur.play_move(&mv);
            let cid = self.ensure_child(node_id, bj);
            path.push(cid);
            node_id = cid;
        }
        let leaf = node_id;
        if self.arena[leaf].is_terminal == -1 {
            if cur.is_terminal_b() {
                let tp = self.arena[leaf].to_play as f64;
                let v = cur.result_white() as f64 * tp;
                self.arena[leaf].is_terminal = 1;
                self.arena[leaf].term_value = v;
            } else {
                self.arena[leaf].is_terminal = 0;
            }
        }
        if self.arena[leaf].is_terminal == 1 {
            let v = self.arena[leaf].term_value;
            self.backup(&path, v);
            self.sims += 1;
            return;
        }
        // Virtual loss along the descended children (not the root) so subsequent
        // descents this round diverge.
        for &nd in &path[1..] {
            let n = &mut self.arena[nd];
            n.n += 1;
            n.w += 1.0;
        }
        let moves = cur.legal_moves_vec();
        let actions: Vec<i32> = moves.iter().map(|m| cur.action_index(m)).collect();
        let row = batch.len() / BUF;
        let start = batch.len();
        batch.resize(start + BUF, 0.0);
        cur.encode_buf(&mut batch[start..start + BUF]);
        self.pending.push(Pending {
            path,
            leaf_id: leaf,
            row,
            is_root: false,
            leaf_moves: moves,
            leaf_actions: actions,
        });
    }

    /// Emit the root's eval request (one row).
    fn request_root(&mut self, batch: &mut Vec<f32>) {
        let moves = self.root_board.legal_moves_vec();
        let actions: Vec<i32> = moves.iter().map(|m| self.root_board.action_index(m)).collect();
        let row = batch.len() / BUF;
        let start = batch.len();
        batch.resize(start + BUF, 0.0);
        self.root_board.encode_buf(&mut batch[start..start + BUF]);
        self.pending = vec![Pending {
            path: Vec::new(),
            leaf_id: self.root_id,
            row,
            is_root: true,
            leaf_moves: moves,
            leaf_actions: actions,
        }];
    }

    /// Apply the round's network results to this game's pending leaves.
    fn apply_eval(&mut self, logits_s: &[f32], values_s: &[f32]) {
        let pend = std::mem::take(&mut self.pending);
        for p in pend {
            let lrow = &logits_s[p.row * ACTION_SIZE..(p.row + 1) * ACTION_SIZE];
            let value = values_s[p.row] as f64;
            if p.is_root {
                let priors = softmax_gather(lrow, &p.leaf_actions);
                let n_leg = p.leaf_moves.len();
                let nd = &mut self.arena[self.root_id];
                nd.legal_moves = p.leaf_moves;
                nd.legal_actions = p.leaf_actions;
                nd.priors = priors;
                nd.children = vec![-1; n_leg];
                nd.expanded = true;
                nd.is_terminal = 0;
                nd.n = 1;
                nd.w = value;
                self.phase = Phase::Running;
            } else {
                // Remove virtual loss along the descended children.
                for &nd in &p.path[1..] {
                    let n = &mut self.arena[nd];
                    n.n -= 1;
                    n.w -= 1.0;
                }
                if !self.arena[p.leaf_id].expanded {
                    let priors = softmax_gather(lrow, &p.leaf_actions);
                    let n_leg = p.leaf_moves.len();
                    let leaf = &mut self.arena[p.leaf_id];
                    leaf.legal_moves = p.leaf_moves;
                    leaf.legal_actions = p.leaf_actions;
                    leaf.priors = priors;
                    leaf.children = vec![-1; n_leg];
                    leaf.expanded = true;
                    leaf.is_terminal = 0;
                }
                self.backup(&p.path, value);
                self.sims += 1;
            }
        }
    }

    fn best_action_move(&self) -> (i32, shakmaty::Move) {
        let root = &self.arena[self.root_id];
        let mut best_n: i64 = -1;
        let mut bj = 0usize;
        for j in 0..root.legal_moves.len() {
            let cid = root.children[j];
            let n = if cid < 0 {
                0u32
            } else {
                self.arena[cid as usize].n
            };
            if (n as i64) > best_n {
                best_n = n as i64;
                bj = j;
            }
        }
        (root.legal_actions[bj], root.legal_moves[bj].clone())
    }
}

#[inline]
fn alive(g: &Game, max_ply: u32) -> bool {
    !g.finished && !g.board.is_terminal_b() && g.ply < max_ply
}

fn result_white(g: &Game, mat_thresh: f64) -> i32 {
    if g.board.is_terminal_b() {
        g.board.result_white()
    } else {
        let d = g.board.material_white() as f64;
        if d >= mat_thresh {
            1
        } else if d <= -mat_thresh {
            -1
        } else {
            0
        }
    }
}

/// Call eval_fn(planes (M,18,8,8)) -> (logits (M,4672), values (M,)).
fn call_eval<'py>(
    py: Python<'py>,
    eval_fn: &PyObject,
    batch: Vec<f32>,
    m: usize,
) -> PyResult<(Vec<f32>, Vec<f32>)> {
    let planes = PyArray1::from_vec_bound(py, batch).reshape([m, 18, 8, 8])?;
    let res = eval_fn.call1(py, (planes,))?;
    let bound = res.bind(py);
    let tup = bound.downcast::<PyTuple>()?;
    let logits: PyReadonlyArray2<f32> = tup.get_item(0)?.extract()?;
    let values: PyReadonlyArray1<f32> = tup.get_item(1)?.extract()?;
    Ok((logits.as_slice()?.to_vec(), values.as_slice()?.to_vec()))
}

/// One opening ply for all active games: sample a move from the to-move net's
/// legal-softmax policy (per-game RNG) -> game variety.
fn opening_ply(
    py: Python<'_>,
    games: &mut [Game],
    idxs: &[usize],
    eval_fn: &PyObject,
    max_ply: u32,
) -> PyResult<()> {
    let mut batch = vec![0f32; idxs.len() * BUF];
    let mut legals: Vec<Vec<i32>> = Vec::with_capacity(idxs.len());
    for (row, &gi) in idxs.iter().enumerate() {
        let g = &games[gi];
        let moves = g.board.legal_moves_vec();
        let acts: Vec<i32> = moves.iter().map(|m| g.board.action_index(m)).collect();
        g.board.encode_buf(&mut batch[row * BUF..(row + 1) * BUF]);
        legals.push(acts);
    }
    let (logits_s, _values) = call_eval(py, eval_fn, batch, idxs.len())?;
    for (row, &gi) in idxs.iter().enumerate() {
        let lrow = &logits_s[row * ACTION_SIZE..(row + 1) * ACTION_SIZE];
        let probs = softmax_gather(lrow, &legals[row]);
        let r = games[gi].rng.next_f64();
        let mut acc = 0.0;
        let mut pick = probs.len() - 1;
        for (i, &p) in probs.iter().enumerate() {
            acc += p;
            if r < acc {
                pick = i;
                break;
            }
        }
        let moves = games[gi].board.legal_moves_vec();
        games[gi].board.play_move(&moves[pick]);
        games[gi].ply += 1;
        let _ = max_ply;
    }
    Ok(())
}

/// Play `k` games in lockstep between net A and net B (A=White iff a_is_white).
/// Returns (results_white, sims_a, sims_b, moves) and accumulates timing.
#[allow(clippy::too_many_arguments)]
fn play_half(
    py: Python<'_>,
    eval_fn_a: &PyObject,
    eval_fn_b: &PyObject,
    k: usize,
    a_is_white: bool,
    ms_per_move: f64,
    budget_a: i64,
    budget_b: i64,
    l_a: usize,
    l_b: usize,
    c_puct: f64,
    max_ply: u32,
    mat_thresh: f64,
    open_plies: u32,
    seed: u64,
    half_id: u64,
    record_moves: bool,
    tree_s: &mut f64,
    eval_s: &mut f64,
    rounds_tot: &mut u64,
    sims_a: &mut Vec<u64>,
    sims_b: &mut Vec<u64>,
) -> PyResult<(Vec<i32>, Vec<Vec<i32>>)> {
    let mut games: Vec<Game> = (0..k)
        .map(|i| {
            Game::new(
                Board::start_pos(),
                GameRng::seeded(seed.wrapping_add(half_id.wrapping_mul(0x9E37_79B9)), i as u64),
            )
        })
        .collect();

    // Opening plies (policy-sampled, batched per ply).
    for op in 0..open_plies {
        let idxs: Vec<usize> = (0..k).filter(|&i| alive(&games[i], max_ply)).collect();
        if idxs.is_empty() {
            break;
        }
        let white = (op % 2) == 0; // ply parity (all games synced)
        let net_is_a = white == a_is_white;
        let ef = if net_is_a { eval_fn_a } else { eval_fn_b };
        opening_ply(py, &mut games, &idxs, ef, max_ply)?;
    }

    // Search moves.
    loop {
        let idxs: Vec<usize> = (0..k).filter(|&i| alive(&games[i], max_ply)).collect();
        if idxs.is_empty() {
            break;
        }
        let white = (games[idxs[0]].ply % 2) == 0;
        let net_is_a = white == a_is_white;
        let ef = if net_is_a { eval_fn_a } else { eval_fn_b };
        let l = if net_is_a { l_a } else { l_b };
        let budget = if net_is_a { budget_a } else { budget_b };
        let time_mode = budget_a <= 0 && budget_b <= 0;

        for &gi in &idxs {
            games[gi].start_search();
        }

        let t0 = Instant::now();
        let mut running_rounds = 0u64;
        loop {
            let was_root = games[idxs[0]].phase == Phase::NeedRoot;
            // Build the round's batch (Rust tree work).
            let tt = Instant::now();
            let mut batch: Vec<f32> = Vec::new();
            for &gi in &idxs {
                if games[gi].phase == Phase::NeedRoot {
                    games[gi].request_root(&mut batch);
                } else {
                    games[gi].pending.clear();
                    for _ in 0..l {
                        games[gi].descend_one(c_puct, &mut batch);
                    }
                }
            }
            let m = batch.len() / BUF;
            *tree_s += tt.elapsed().as_secs_f64();

            // One batched forward (Python).
            let te = Instant::now();
            let (logits_s, values_s) = if m > 0 {
                call_eval(py, ef, batch, m)?
            } else {
                (Vec::new(), Vec::new())
            };
            *eval_s += te.elapsed().as_secs_f64();

            // Scatter results (Rust tree work).
            let tb = Instant::now();
            for &gi in &idxs {
                games[gi].apply_eval(&logits_s, &values_s);
            }
            *tree_s += tb.elapsed().as_secs_f64();

            if was_root {
                continue; // root-expand round; not a sim round
            }
            running_rounds += 1;
            *rounds_tot += 1;
            if !time_mode {
                if idxs.iter().all(|&gi| games[gi].sims >= budget as u64) {
                    break;
                }
            } else if t0.elapsed().as_secs_f64() * 1000.0 >= ms_per_move && running_rounds >= 1 {
                break;
            }
        }

        // Play argmax-visit moves.
        for &gi in &idxs {
            let (a, mv) = games[gi].best_action_move();
            if net_is_a {
                sims_a.push(games[gi].sims);
            } else {
                sims_b.push(games[gi].sims);
            }
            if record_moves {
                games[gi].moves.push(a);
            }
            games[gi].board.play_move(&mv);
            games[gi].ply += 1;
            if games[gi].board.is_terminal_b() || games[gi].ply >= max_ply {
                games[gi].finished = true;
            }
        }
    }

    let results: Vec<i32> = games.iter().map(|g| result_white(g, mat_thresh)).collect();
    let moves: Vec<Vec<i32>> = if record_moves {
        games.iter().map(|g| g.moves.clone()).collect()
    } else {
        Vec::new()
    };
    Ok((results, moves))
}

fn score_a(results: &[i32], a_is_white: bool) -> f64 {
    let mut s = 0.0;
    for &r in results {
        if a_is_white {
            s += if r > 0 { 1.0 } else if r == 0 { 0.5 } else { 0.0 };
        } else {
            s += if r < 0 { 1.0 } else if r == 0 { 0.5 } else { 0.0 };
        }
    }
    s
}

/// Full match between two nets, colors balanced. PER-SIDE fixed sims: side A
/// searches `sims_a` sims/move, side B `sims_b` (each at its own L) -- so configs
/// trained at different sims can be evaluated at their own operating point. If
/// both sims_a<=0 and sims_b<=0, the move is time-budgeted by `ms_per_move`.
/// Returns (score_a, stats_dict).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (eval_fn_a, eval_fn_b, n_games, ms_per_move, sims_a, sims_b, l_a, l_b,
                    c_puct, max_ply, mat_thresh, open_plies, seed, record_moves))]
pub fn arena_match(
    py: Python<'_>,
    eval_fn_a: PyObject,
    eval_fn_b: PyObject,
    n_games: usize,
    ms_per_move: f64,
    sims_a: i64,
    sims_b: i64,
    l_a: usize,
    l_b: usize,
    c_puct: f64,
    max_ply: u32,
    mat_thresh: f64,
    open_plies: u32,
    seed: u64,
    record_moves: bool,
) -> PyResult<(f64, PyObject)> {
    let mut tree_s = 0.0;
    let mut eval_s = 0.0;
    let mut rounds_tot = 0u64;
    let mut sims_used_a: Vec<u64> = Vec::new();
    let mut sims_used_b: Vec<u64> = Vec::new();
    let mut all_results: Vec<i32> = Vec::new();
    let mut all_moves: Vec<Vec<i32>> = Vec::new();

    let ka = n_games / 2;
    let kb = n_games - ka;
    let mut score = 0.0;

    for (half_id, (k, a_is_white)) in [(ka, true), (kb, false)].into_iter().enumerate() {
        if k == 0 {
            continue;
        }
        let (results, moves) = play_half(
            py, &eval_fn_a, &eval_fn_b, k, a_is_white, ms_per_move, sims_a, sims_b, l_a, l_b,
            c_puct, max_ply, mat_thresh, open_plies, seed, half_id as u64, record_moves,
            &mut tree_s, &mut eval_s, &mut rounds_tot, &mut sims_used_a, &mut sims_used_b,
        )?;
        score += score_a(&results, a_is_white);
        all_results.extend(results);
        all_moves.extend(moves);
    }

    let d = PyDict::new_bound(py);
    d.set_item("tree_s", tree_s)?;
    d.set_item("eval_s", eval_s)?;
    d.set_item("rounds", rounds_tot)?;
    d.set_item("sims_a", sims_used_a)?;
    d.set_item("sims_b", sims_used_b)?;
    d.set_item("results_white", all_results)?;
    if record_moves {
        let ml = PyList::empty_bound(py);
        for mv in all_moves {
            ml.append(mv)?;
        }
        d.set_item("moves", ml)?;
    }
    Ok((score, d.into_any().unbind()))
}

/// Play a fixed list of games, each starting from its own board with its own
/// A=White flag. Unlike `play_half`, games may be at DIFFERENT positions (and
/// thus different sides to move), so each round groups pending leaves by the
/// searching net into TWO batched forwards (one per net). No opening sampling
/// here -- the book FEN provides the diversity.
#[allow(clippy::too_many_arguments)]
fn play_games_grouped(
    py: Python<'_>,
    ef_a: &PyObject,
    ef_b: &PyObject,
    mut games: Vec<Game>,
    a_white: &[bool],
    ms_per_move: f64,
    budget_a: i64,
    budget_b: i64,
    l_a: usize,
    l_b: usize,
    c_puct: f64,
    max_ply: u32,
    mat_thresh: f64,
    record_moves: bool,
    tree_s: &mut f64,
    eval_s: &mut f64,
    rounds_tot: &mut u64,
    sims_used_a: &mut Vec<u64>,
    sims_used_b: &mut Vec<u64>,
) -> PyResult<(Vec<i32>, Vec<Vec<i32>>)> {
    let time_mode = budget_a <= 0 && budget_b <= 0;
    let n = games.len();
    loop {
        let active: Vec<usize> = (0..n).filter(|&i| alive(&games[i], max_ply)).collect();
        if active.is_empty() {
            break;
        }
        for &gi in &active {
            games[gi].start_search();
        }

        let t0 = Instant::now();
        let mut running_rounds = 0u64;
        loop {
            let was_root = games[active[0]].phase == Phase::NeedRoot;
            let tt = Instant::now();
            let mut batch_a: Vec<f32> = Vec::new();
            let mut batch_b: Vec<f32> = Vec::new();
            for &gi in &active {
                let net_is_a = (games[gi].root_board.to_play_i8() == 1) == a_white[gi];
                let bud = if net_is_a { budget_a } else { budget_b };
                if bud > 0 && games[gi].sims >= bud as u64 {
                    continue; // this game already hit its (per-side) sim budget
                }
                let l = if net_is_a { l_a } else { l_b };
                if games[gi].phase == Phase::NeedRoot {
                    if net_is_a {
                        games[gi].request_root(&mut batch_a);
                    } else {
                        games[gi].request_root(&mut batch_b);
                    }
                } else {
                    games[gi].pending.clear();
                    for _ in 0..l {
                        if net_is_a {
                            games[gi].descend_one(c_puct, &mut batch_a);
                        } else {
                            games[gi].descend_one(c_puct, &mut batch_b);
                        }
                    }
                }
            }
            let ma = batch_a.len() / BUF;
            let mb = batch_b.len() / BUF;
            *tree_s += tt.elapsed().as_secs_f64();

            let te = Instant::now();
            let (la, va) = if ma > 0 {
                call_eval(py, ef_a, batch_a, ma)?
            } else {
                (Vec::new(), Vec::new())
            };
            let (lb, vb) = if mb > 0 {
                call_eval(py, ef_b, batch_b, mb)?
            } else {
                (Vec::new(), Vec::new())
            };
            *eval_s += te.elapsed().as_secs_f64();

            let tb = Instant::now();
            for &gi in &active {
                let net_is_a = (games[gi].root_board.to_play_i8() == 1) == a_white[gi];
                let bud = if net_is_a { budget_a } else { budget_b };
                if bud > 0 && games[gi].sims >= bud as u64 {
                    continue;
                }
                if net_is_a {
                    games[gi].apply_eval(&la, &va);
                } else {
                    games[gi].apply_eval(&lb, &vb);
                }
            }
            *tree_s += tb.elapsed().as_secs_f64();

            if was_root {
                continue;
            }
            running_rounds += 1;
            *rounds_tot += 1;
            if !time_mode {
                if active.iter().all(|&gi| {
                    let net_is_a = (games[gi].root_board.to_play_i8() == 1) == a_white[gi];
                    let bud = if net_is_a { budget_a } else { budget_b };
                    games[gi].sims >= bud as u64
                }) {
                    break;
                }
            } else if t0.elapsed().as_secs_f64() * 1000.0 >= ms_per_move && running_rounds >= 1 {
                break;
            }
        }

        for &gi in &active {
            let net_is_a = (games[gi].root_board.to_play_i8() == 1) == a_white[gi];
            if net_is_a {
                sims_used_a.push(games[gi].sims);
            } else {
                sims_used_b.push(games[gi].sims);
            }
            let (a, mv) = games[gi].best_action_move();
            if record_moves {
                games[gi].moves.push(a);
            }
            games[gi].board.play_move(&mv);
            games[gi].ply += 1;
            if games[gi].board.is_terminal_b() || games[gi].ply >= max_ply {
                games[gi].finished = true;
            }
        }
    }

    let results: Vec<i32> = games.iter().map(|g| result_white(g, mat_thresh)).collect();
    let moves: Vec<Vec<i32>> = if record_moves {
        games.iter().map(|g| g.moves.clone()).collect()
    } else {
        Vec::new()
    };
    Ok((results, moves))
}

/// Tournament match over a FIXED opening suite. Each FEN in `opening_fens` is
/// played BOTH ways (A=White and B=White) `games_per_opening_pair` times, with
/// deterministic best-play from the book position (no opening sampling). Colors
/// are balanced by construction. Returns (score_a, stats).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (eval_fn_a, eval_fn_b, opening_fens, games_per_opening_pair,
                    ms_per_move, sims_a, sims_b, l_a, l_b, c_puct, max_ply, mat_thresh,
                    seed, record_moves))]
pub fn arena_match_openings(
    py: Python<'_>,
    eval_fn_a: PyObject,
    eval_fn_b: PyObject,
    opening_fens: Vec<String>,
    games_per_opening_pair: usize,
    ms_per_move: f64,
    sims_a: i64,
    sims_b: i64,
    l_a: usize,
    l_b: usize,
    c_puct: f64,
    max_ply: u32,
    mat_thresh: f64,
    seed: u64,
    record_moves: bool,
) -> PyResult<(f64, PyObject)> {
    let mut games: Vec<Game> = Vec::new();
    let mut a_white: Vec<bool> = Vec::new();
    let mut idx = 0u64;
    for fen in &opening_fens {
        for _ in 0..games_per_opening_pair {
            for &aw in &[true, false] {
                let board = Board::from_fen(fen)?;
                games.push(Game::new(board, GameRng::seeded(seed, idx)));
                a_white.push(aw);
                idx += 1;
            }
        }
    }

    let mut tree_s = 0.0;
    let mut eval_s = 0.0;
    let mut rounds_tot = 0u64;
    let mut sims_used_a: Vec<u64> = Vec::new();
    let mut sims_used_b: Vec<u64> = Vec::new();
    let (results, moves) = play_games_grouped(
        py, &eval_fn_a, &eval_fn_b, games, &a_white, ms_per_move, sims_a, sims_b, l_a, l_b,
        c_puct, max_ply, mat_thresh, record_moves, &mut tree_s, &mut eval_s, &mut rounds_tot,
        &mut sims_used_a, &mut sims_used_b,
    )?;

    let mut score = 0.0;
    let mut a_w_count = 0usize;
    for (i, &r) in results.iter().enumerate() {
        if a_white[i] {
            a_w_count += 1;
            score += if r > 0 { 1.0 } else if r == 0 { 0.5 } else { 0.0 };
        } else {
            score += if r < 0 { 1.0 } else if r == 0 { 0.5 } else { 0.0 };
        }
    }

    let d = PyDict::new_bound(py);
    d.set_item("tree_s", tree_s)?;
    d.set_item("eval_s", eval_s)?;
    d.set_item("rounds", rounds_tot)?;
    d.set_item("n_games", results.len())?;
    d.set_item("a_white_count", a_w_count)?;
    d.set_item("b_white_count", results.len() - a_w_count)?;
    d.set_item("sims_a", sims_used_a)?;
    d.set_item("sims_b", sims_used_b)?;
    d.set_item("results_white", results)?;
    d.set_item("a_white", a_white.to_vec())?;
    if record_moves {
        let ml = PyList::empty_bound(py);
        for mv in moves {
            ml.append(mv)?;
        }
        d.set_item("moves", ml)?;
    }
    Ok((score, d.into_any().unbind()))
}

/// Single-game, single-net throughput probe: play ONE move-step from the start
/// position under the given budget and return (sims, rounds, tree_s, eval_s,
/// wall_s). Used for the nodes/s + GPU-forward-vs-Rust-tree split sweep.
#[pyfunction]
#[pyo3(signature = (eval_fn, ms_per_move, fixed_sims, l, c_puct, seed))]
pub fn arena_bench(
    py: Python<'_>,
    eval_fn: PyObject,
    ms_per_move: f64,
    fixed_sims: i64,
    l: usize,
    c_puct: f64,
    seed: u64,
) -> PyResult<(u64, u64, f64, f64, f64)> {
    let mut g = Game::new(Board::start_pos(), GameRng::seeded(seed, 0));
    g.start_search();
    let mut tree_s = 0.0;
    let mut eval_s = 0.0;
    let mut rounds = 0u64;
    let t0 = Instant::now();
    loop {
        let was_root = g.phase == Phase::NeedRoot;
        let tt = Instant::now();
        let mut batch: Vec<f32> = Vec::new();
        if g.phase == Phase::NeedRoot {
            g.request_root(&mut batch);
        } else {
            g.pending.clear();
            for _ in 0..l {
                g.descend_one(c_puct, &mut batch);
            }
        }
        let m = batch.len() / BUF;
        tree_s += tt.elapsed().as_secs_f64();

        let te = Instant::now();
        let (logits_s, values_s) = if m > 0 {
            call_eval(py, &eval_fn, batch, m)?
        } else {
            (Vec::new(), Vec::new())
        };
        eval_s += te.elapsed().as_secs_f64();

        let tb = Instant::now();
        g.apply_eval(&logits_s, &values_s);
        tree_s += tb.elapsed().as_secs_f64();

        if was_root {
            continue;
        }
        rounds += 1;
        if fixed_sims > 0 {
            if g.sims >= fixed_sims as u64 {
                break;
            }
        } else if t0.elapsed().as_secs_f64() * 1000.0 >= ms_per_move {
            break;
        }
    }
    Ok((g.sims, rounds, tree_s, eval_s, t0.elapsed().as_secs_f64()))
}

/// Search ONE position (for interactive human-vs-net play) with the SAME
/// leaf-parallel PUCT as the arena: `sims` total sims, leaf-batch `l`, `c_puct`,
/// deterministic best-play (argmax visits, legal-order tie-break). `eval_fn` has
/// the arena signature (planes (M,18,8,8) f32 -> (logits (M,4672) f32, values
/// (M,) f32)). Returns (best_action, best_uci, root_value): best_action = the
/// canonical 4672 action index (== chess_env.encode_move), best_uci = the
/// python-chess-compatible UCI string, root_value = root mean value from the
/// side-to-move's perspective (for display).
#[pyfunction]
#[pyo3(signature = (eval_fn, fen, sims, l, c_puct))]
pub fn search_position(
    py: Python<'_>,
    eval_fn: PyObject,
    fen: String,
    sims: i64,
    l: usize,
    c_puct: f64,
) -> PyResult<(i32, String, f64)> {
    let board = Board::from_fen(&fen)?;
    if board.is_terminal_b() || board.legal_moves_vec().is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "no legal moves (terminal position)",
        ));
    }
    let mut g = Game::new(board, GameRng::seeded(0, 0));
    g.start_search();
    loop {
        let was_root = g.phase == Phase::NeedRoot;
        let mut batch: Vec<f32> = Vec::new();
        if g.phase == Phase::NeedRoot {
            g.request_root(&mut batch);
        } else {
            g.pending.clear();
            for _ in 0..l.max(1) {
                g.descend_one(c_puct, &mut batch);
            }
        }
        let m = batch.len() / BUF;
        let (logits_s, values_s) = if m > 0 {
            call_eval(py, &eval_fn, batch, m)?
        } else {
            (Vec::new(), Vec::new())
        };
        g.apply_eval(&logits_s, &values_s);
        if was_root {
            continue; // root-expand round; not a sim round
        }
        if sims > 0 && g.sims >= sims as u64 {
            break;
        }
    }
    let (action, mv) = g.best_action_move();
    let uci = crate::move_uci(&mv);
    let root = &g.arena[g.root_id];
    let root_value = if root.n > 0 { root.w / root.n as f64 } else { 0.0 };
    Ok((action, uci, root_value))
}
