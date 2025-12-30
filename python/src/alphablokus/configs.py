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
    model_class: str
    main_body_channels: int
    residual_blocks: int
    value_head_channels: int
    value_head_flat_layer_width: int
    policy_head_channels: int
    policy_convolution_kernel: int

    def __init__(self, config_file: str):
        with open(config_file, "rb") as f:
            data = tomllib.load(f)

        self.model_class = data["network"]["model_class"]
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
    ignore_invalid_moves: bool
    sampling_ratio: float
    window_size: int
    device: str
    simulated: bool
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
        self.ignore_invalid_moves = training_data["ignore_invalid_moves"]
        self.sampling_ratio = training_data["sampling_ratio"]
        self.window_size = training_data["window_size"]
        self.device = training_data["device"]
        self.simulated = training_data["simulated"]
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


@dataclass
class TrainingStandaloneConfig:
    device: str
    learning_rate: float
    policy_loss_weight: float
    ignore_invalid_moves: bool
    num_epochs: int
    batch_size: int
    shuffle_buffer_file_count: int
    train_batches_per_test: int
    num_workers: int
    prefetch_factor: int
    remote_train_data_dir: str
    remote_test_data_dir: str
    local_game_mirror: str
    aim_repo_path: str

    def __init__(self, config_file: str):
        with open(config_file, "rb") as f:
            data = tomllib.load(f)

        standalone_data = data["training_standalone"]

        self.device = standalone_data["device"]
        self.learning_rate = standalone_data["learning_rate"]
        self.policy_loss_weight = standalone_data["policy_loss_weight"]
        self.ignore_invalid_moves = standalone_data["ignore_invalid_moves"]
        self.num_epochs = standalone_data["num_epochs"]
        self.batch_size = standalone_data["batch_size"]
        self.shuffle_buffer_file_count = standalone_data["shuffle_buffer_file_count"]
        self.train_batches_per_test = standalone_data["train_batches_per_test"]
        self.num_workers = standalone_data["num_workers"]
        self.prefetch_factor = standalone_data["prefetch_factor"]
        self.remote_train_data_dir = standalone_data["remote_train_data_dir"]
        self.remote_test_data_dir = standalone_data["remote_test_data_dir"]
        self.local_game_mirror = standalone_data["local_game_mirror"]
        self.aim_repo_path = standalone_data["aim_repo_path"]
