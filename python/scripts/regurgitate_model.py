config_path = "/Users/shivamsarodia/Dev/AlphaBlokus/configs/train_live/local_vast.toml"
initial_training_path = "s3://alpha-blokus/full_v2/training_simulated/v5_19m_to_10m_windowed/base_no_opt_init_1770216462.pth"

output_training_directory = "s3://alpha-blokus/full_v2/training_v5/"
output_model_directory = "s3://alpha-blokus/full_v2/models_v5/"
output_name = "018000000"

clear_optimizer = True

from alphablokus.configs import GameConfig, NetworkConfig, TrainingLiveConfig
from alphablokus.train_utils import load_initial_state, save_model_and_state

game_config = GameConfig(config_path)
network_config = NetworkConfig(config_path)
training_config = TrainingLiveConfig(config_path)

model, optimizer = load_initial_state(
    network_config,
    game_config,
    learning_rate=training_config.learning_rate,
    device=training_config.device,
    training_file=initial_training_path,
    skip_loading_optimizer=clear_optimizer,
)

save_model_and_state(
    model=model,
    optimizer=optimizer,
    name=output_name,
    model_directory=output_model_directory,
    training_directory=output_training_directory,
    device=training_config.device,
)
