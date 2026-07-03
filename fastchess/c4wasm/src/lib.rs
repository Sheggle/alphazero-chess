//! WebAssembly bindings for the Score-Four alpha-beta engine.
//!
//! Reuses the SAME pure-Rust core as the native pyo3 build (`../src/connect4.rs`)
//! via a `#[path]` module include, so the browser engine is bit-identical to the
//! native one. The only platform detail — the search deadline — is provided as a
//! `Date.now()` closure (time-based) with a node-budget fallback.

#[path = "../../src/connect4.rs"]
mod connect4;

use wasm_bindgen::prelude::*;

/// Result of a search, exposed to JS as an object `{col, depth, nodes, score}`.
#[wasm_bindgen]
pub struct C4Result {
    pub col: i32,
    pub depth: i32,
    pub nodes: f64,
    pub score: f64,
}

fn pack(r: (i32, i32, u64, i32)) -> C4Result {
    C4Result { col: r.0, depth: r.1, nodes: r.2 as f64, score: connect4::score_to_f64(r.3) }
}

/// Iterative-deepening search with a wall-clock budget (`Date.now()`).
///
/// `board`: 64 flat cells (`cell = x*16 + y*4 + z`), values {0,+1,-1}.
/// `to_play`: +1 / -1. `w1,w2`: line weights (W3 fixed at 1.0).
#[wasm_bindgen]
pub fn c4_best_move(board: &[i8], to_play: i8, time_ms: u32, w1: f32, w2: f32) -> C4Result {
    if board.len() != 64 {
        return C4Result { col: -1, depth: 0, nodes: 0.0, score: 0.0 };
    }
    let mut eng = connect4::C4::from_board(board, to_play, w1 as f64, w2 as f64);
    let deadline = js_sys::Date::now() + time_ms as f64;
    eng.set_stop_fn(Box::new(move |_nodes| js_sys::Date::now() >= deadline));
    pack(eng.search_id())
}

/// Iterative-deepening search bounded by a NODE budget (deterministic; no clock).
/// Handy when the caller measured n/s once and prefers a portable budget.
#[wasm_bindgen]
pub fn c4_best_move_nodes(board: &[i8], to_play: i8, node_budget: f64, w1: f32, w2: f32) -> C4Result {
    if board.len() != 64 {
        return C4Result { col: -1, depth: 0, nodes: 0.0, score: 0.0 };
    }
    let mut eng = connect4::C4::from_board(board, to_play, w1 as f64, w2 as f64);
    let budget = node_budget as u64;
    eng.set_stop_fn(Box::new(move |nodes| nodes >= budget));
    pack(eng.search_id())
}

/// Fixed-depth search (deterministic, no clock).
#[wasm_bindgen]
pub fn c4_best_move_depth(board: &[i8], to_play: i8, depth: i32, w1: f32, w2: f32) -> C4Result {
    if board.len() != 64 {
        return C4Result { col: -1, depth: 0, nodes: 0.0, score: 0.0 };
    }
    let mut eng = connect4::C4::from_board(board, to_play, w1 as f64, w2 as f64);
    pack(eng.search_depth(depth))
}

/// Depth-0 static heuristic in the side-to-move frame.
#[wasm_bindgen]
pub fn c4_eval(board: &[i8], to_play: i8, w1: f32, w2: f32) -> f64 {
    if board.len() != 64 {
        return 0.0;
    }
    connect4::C4::from_board(board, to_play, w1 as f64, w2 as f64).static_eval() as f64
        / connect4::SCALE as f64
}
