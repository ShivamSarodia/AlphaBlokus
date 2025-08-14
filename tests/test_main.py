from hydra import initialize, compose

from alpha_blokus.__main__ import main

def test_selfplay_loop():
    with initialize(version_base=None, config_path="../configs/"):
        cfg = compose(config_name="main_self_play")

    main(cfg)
