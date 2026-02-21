# Key S3 directories

- s3://alpha-blokus/full_v2/models_simulated -- where onnx files for simulated runs are generally saved, sometimes in a subdirectory inside. note that usually each onnx file also has an onnx.data file
- s3://alpha-blokus/full_v2/training_simulated -- where the pth files for simulated runs are generally saved, sometimes in a subdirectory inside. the file name in this directory is the same as the file name in the models_simulated directory, just the extensions differ.
- s3://alpha-blokus/full_v2/games -- where most game data is stored. the format of game data files is like 2025-12-29_17-35-09-188900145-93928152_10004.bin where the final `10004` is the number of samples in that file. (Just FYI - early on, most files have approximately 10,000 samples, and the later on, most files have approximately 2,000 samples.)

# Creating config files

- When I ask you to create a config for playing against Pentobi, model your generated file off of configs/self_play/v_pentobi.toml. You should use the S3 model path to an onnx file I provide (and ask for one if I don't give you one). The name of each agent should be {model}_mcts and {model}_pentobi where {model} is something that allows me to identify the published metrics afterwards. Often, the file name of the S3 model path is ideal.


# Commands

- Sometimes I will ask you to generate a command I can run to do self-play. That command takes the form of `cargo run --release --bin self-play -- --config configs/...`.
