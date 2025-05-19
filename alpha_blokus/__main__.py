import os
import numpy as np
import hydra

from alpha_blokus import simulation
from alpha_blokus.moves_data import moves_data
from alpha_blokus.training import standalone
from alpha_blokus.training import unlooped
@hydra.main(version_base=None, config_path="../conf/")
def main(cfg):
    if cfg["entrypoint"] == "simulation":
        simulation.run(cfg)
    elif cfg["entrypoint"] == "standalone_training":
        standalone.run(cfg)
    elif cfg["entrypoint"] == "unlooped_training":
        unlooped.run(cfg)

if __name__ == "__main__":
    main() 