learning_rate = 1e-3
config_path = "/Users/shivamsarodia/Dev/AlphaBlokus/configs/train_offline/with_film.toml"
initial_training_path = "s3://alpha-blokus/full_v3/training_simulated/fabric/010000000_1772621499.pth"

output_training_directory = "s3://alpha-blokus/full_v3/training/"
output_model_directory = "s3://alpha-blokus/full_v3/models/"
output_name = "010000000"

clear_optimizer = True

from alphablokus.configs import GameConfig, NetworkConfig
from alphablokus.train_utils import load_initial_state, save_model_and_state

game_config = GameConfig(config_path)
network_config = NetworkConfig(config_path)

model, optimizer = load_initial_state(
    network_config,
    game_config,
    learning_rate=learning_rate,
    device="cpu",
    training_file=initial_training_path,
    skip_loading_optimizer=clear_optimizer,
)

save_model_and_state(
    model=model,
    optimizer=optimizer,
    name=output_name,
    model_directory=output_model_directory,
    training_directory=output_training_directory,
    device="cpu",
)
