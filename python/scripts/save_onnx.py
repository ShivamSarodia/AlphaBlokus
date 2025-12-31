import sys
from alphablokus.configs import GameConfig, NetworkConfig
from alphablokus.train_utils import initialize_model
from alphablokus.files import from_localized

config_path = sys.argv[1]
game_config = GameConfig(config_path)
network_config = NetworkConfig(config_path)

onnx_path = input("Enter the path to save the ONNX model: ")
model = initialize_model(network_config, game_config)
with from_localized(onnx_path) as onnx_path:
    model.save_onnx(onnx_path, "cpu")
