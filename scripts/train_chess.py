import sys
from alphazero.chess_train import ChessConfig, train


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "overnight"
    if mode == "smoke":
        cfg = ChessConfig(iterations=3, games_per_iter=4, n_workers=2, sims=16,
                          max_ply=40, train_steps=30, eval_every=2, eval_games=6,
                          ckpt_dir="models/chess_smoke")
    else:
        # Tuned keeper config (CHESS_LOG): wider candidate set (chess width),
        # softer targets, lower lr (value overfit), entropy exploration.
        cfg = ChessConfig(
            channels=64, blocks=6,
            iterations=100000,            # runs until stopped; checkpoints every iter
            games_per_iter=28, n_workers=7, train_threads=4,
            sims=32, max_considered=16, max_ply=80,
            c_visit=50.0, c_scale=0.3, mat_thresh=1.0,
            buffer_size=150000, batch_size=256, train_steps=120,
            lr=1e-3, weight_decay=1e-4, entropy_coef=0.05,
            eval_every=8, eval_games=16,
            ckpt_dir="models/chess_keeper", seed=0,
        )
    train(cfg)


if __name__ == "__main__":
    main()
