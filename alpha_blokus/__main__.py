import hydra
from omegaconf import DictConfig

from alpha_blokus.entrypoints import selfplay_loop


@hydra.main(version_base=None, config_path="../configs/")
def main(cfg: DictConfig):
    if cfg["entrypoint"] == "selfplay_loop":
        selfplay_loop.run(cfg)


if __name__ == "__main__":
    main()
