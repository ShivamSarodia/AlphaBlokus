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
class TrainingLiveConfig:
    learning_rate: float
    batch_size: int
    policy_loss_weight: float
    sampling_ratio: float
    window_size: int
    device: str
    min_samples_for_save: int
    poll_interval_seconds: int
    in_memory_shuffle_file_count: int
    num_workers: int
    prefetch_factor: int
    local_cache_dir: str
    cleanup_local_files: bool
    game_data_directory: str
    model_directory: str
    training_directory: str

    def __init__(self, config_file: str):
        with open(config_file, "rb") as f:
            data = tomllib.load(f)

        training_data = data["training_live"]
        self.learning_rate = training_data["learning_rate"]
        self.batch_size = training_data["batch_size"]
        self.policy_loss_weight = training_data["policy_loss_weight"]
        self.sampling_ratio = training_data["sampling_ratio"]
        self.window_size = training_data["window_size"]
        self.device = training_data["device"]
        self.min_samples_for_save = training_data["min_samples_for_save"]
        self.poll_interval_seconds = training_data["poll_interval_seconds"]
        self.in_memory_shuffle_file_count = training_data[
            "in_memory_shuffle_file_count"
        ]
        self.num_workers = training_data["num_workers"]
        self.prefetch_factor = training_data["prefetch_factor"]
        self.local_cache_dir = training_data["local_cache_dir"]
        self.cleanup_local_files = training_data["cleanup_local_files"]
        directories_data = data["directories"]
        self.game_data_directory = directories_data["game_data_directory"]
        self.model_directory = directories_data["model_directory"]
        self.training_directory = directories_data["training_directory"]


@dataclass
class TrainingOfflineConfig:
    device: str
    learning_rate: float
    policy_loss_weight: float
    batch_size: int
    in_memory_shuffle_file_count: int
    num_workers: int
    prefetch_factor: int
    game_data_directory: str
    local_game_mirror: str
    model_directory: str
    training_directory: str
    output_name: str
    initial_training_state_file: str
    load_optimizer_from_initial_training_state: bool
    optimizer_type: str

    def __init__(self, config_file: str):
        with open(config_file, "rb") as f:
            data = tomllib.load(f)

        offline_data = data["training_offline"]
        self.device = offline_data["device"]
        self.learning_rate = offline_data["learning_rate"]
        self.policy_loss_weight = offline_data["policy_loss_weight"]
        self.batch_size = offline_data["batch_size"]
        self.in_memory_shuffle_file_count = offline_data["in_memory_shuffle_file_count"]
        self.num_workers = offline_data["num_workers"]
        self.prefetch_factor = offline_data["prefetch_factor"]
        self.game_data_directory = offline_data["game_data_directory"]
        self.local_game_mirror = offline_data["local_game_mirror"]
        self.model_directory = offline_data["model_directory"]
        self.training_directory = offline_data["training_directory"]
        self.output_name = offline_data["output_name"]
        self.initial_training_state_file = offline_data["initial_training_state_file"]
        self.load_optimizer_from_initial_training_state = offline_data[
            "load_optimizer_from_initial_training_state"
        ]
        self.optimizer_type = offline_data["optimizer_type"]
