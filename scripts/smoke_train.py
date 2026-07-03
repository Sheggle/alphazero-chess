from alphazero.train import Config, train
cfg = Config(iterations=3, games_per_iter=15, n_sims=40, train_steps=60, eval_random_games=60)
train(cfg)
