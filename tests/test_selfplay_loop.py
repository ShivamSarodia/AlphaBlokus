import pytest
from omegaconf import OmegaConf

from alpha_blokus.__main__ import main

@pytest.fixture
def base_config():
    return OmegaConf.create({
        "log_to_console": False,
        "entrypoint": "selfplay_loop",
        "gameplay": True,
        "networks": {
            "main": {
                "backend": "torch",
            },
        }
    })

def test_main(base_config):
    cfg = base_config

    del cfg.networks.main
    cfg["networks"]["1"] = {
        "backend": "torch",
    }
    cfg["networks"]["2"] = {
        "backend": "torch",
    }

    with pytest.raises(NotImplementedError):
        main(cfg)
