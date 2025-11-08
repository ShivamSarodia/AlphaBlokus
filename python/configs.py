import tomllib

from dataclasses import dataclass


@dataclass
class GameConfig:
    board_size: int
    num_moves: int
    num_pieces: int
    num_piece_orientations: int

    def __init__(self, config_file: str):
        with open(config_file, "rb") as f:
            data = tomllib.load(f)

        self.board_size = data["game"]["board_size"]
        self.num_moves = data["game"]["num_moves"]
        self.num_pieces = data["game"]["num_pieces"]
        self.num_piece_orientations = data["game"]["num_piece_orientations"]


@dataclass
class NetworkConfig:
    main_body_channels: int
    residual_blocks: int
    value_head_channels: int
    value_head_flat_layer_width: int
    policy_head_channels: int
    policy_convolution_kernel: int

    def __init__(self, config_file: str):
        with open(config_file, "rb") as f:
            data = tomllib.load(f)

        self.main_body_channels = data["network"]["main_body_channels"]
        self.residual_blocks = data["network"]["residual_blocks"]
        self.value_head_channels = data["network"]["value_head_channels"]
        self.value_head_flat_layer_width = data["network"][
            "value_head_flat_layer_width"
        ]
        self.policy_head_channels = data["network"]["policy_head_channels"]
        self.policy_convolution_kernel = data["network"]["policy_convolution_kernel"]


@dataclass
class TrainingConfig:
    num_epochs: int
    learning_rate: float
    batch_size: int
    policy_loss_weight: float
    sampling_ratio: float
    window_size: int
    device: str
    min_samples_for_save: int
    poll_interval_seconds: int

    def __init__(self, config_file: str):
        with open(config_file, "rb") as f:
            data = tomllib.load(f)

        training_data = data["training"]
        self.num_epochs = training_data["num_epochs"]
        self.learning_rate = training_data["learning_rate"]
        self.batch_size = training_data["batch_size"]
        self.policy_loss_weight = training_data["policy_loss_weight"]
        self.sampling_ratio = training_data["sampling_ratio"]
        self.window_size = training_data["window_size"]
        self.device = training_data["device"]
        self.min_samples_for_save = training_data.get("min_samples_for_save", 10000)
        self.poll_interval_seconds = training_data.get("poll_interval_seconds", 60)


@dataclass
class DirectoriesConfig:
    game_data_directory: str
    model_directory: str
    training_directory: str

    def __init__(self, config_file: str):
        with open(config_file, "rb") as f:
            data = tomllib.load(f)

        self.game_data_directory = data["directories"]["game_data_directory"]
        self.model_directory = data["directories"]["model_directory"]
        self.training_directory = data["directories"]["training_directory"]

        assert self.game_data_directory.endswith("/")
        assert self.model_directory.endswith("/") or self.model_directory == ""
        assert self.training_directory.endswith("/") or self.training_directory == ""
