import pytest
from hydra import initialize, compose

from alpha_blokus.__main__ import main

@pytest.mark.skip(reason="Skipping this test until I figure out how to test Ray properly.")
def test_calling_selfplay_loop():
    with initialize(version_base=None, config_path="../configs/"):
        cfg = compose(config_name="main_self_play")

    main(cfg)
