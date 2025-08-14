import pytest
from omegaconf import OmegaConf
from unittest.mock import patch

from alpha_blokus.entrypoints.selfplay_loop import start_inference_actors

@pytest.mark.skip(reason="Skipping tests until I figure out how to test Ray properly.")
def test_start_inference_actors():
    cfg = OmegaConf.create({
        "gameplay": True,
        "networks": {
            "network_a": {
                "backend": "torch",
                "batch_size": 1,
            },
            "network_b": {
                "backend": "torch",
                "batch_size": 1,
            },
        }
    })

    with patch("alpha_blokus.inference.actors.torch.TorchInferenceActor"):
        inference_clients = start_inference_actors(cfg)
        
    assert len(inference_clients) == 2
