import pytest
import random
import os
import hydra

from alpha_blokus.__main__ import main

@pytest.fixture
def cfg():
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(config_name="test")
    return cfg

def test_main(cfg):
    model_dir_path = f"/tmp/alpha_blokus_{random.randint(0, 1000000)}/models/"
    os.makedirs(model_dir_path, exist_ok=True)

    cfg["networks"]["main"]["model_read_path"] = model_dir_path
    main(cfg)

    assert os.path.isfile(os.path.join(model_dir_path, "0.pt"))
