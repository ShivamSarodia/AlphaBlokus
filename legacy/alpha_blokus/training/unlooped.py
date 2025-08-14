import os
import time
import torch
import aim
import random
from torch import nn

from alpha_blokus.neural_net import NeuralNet
from alpha_blokus.training.helpers import GameDataset

def get_validation_losses(model, validation_dataset, cfg):
    results = {
        "total_loss": 0,
        "value_loss": 0,
        "policy_loss": 0,
        "value_max_correct": 0,
        "policy_max_correct": 0,
    }

    total_sample_count = 0

    model.eval()
    with torch.inference_mode():
        while total_sample_count < validation_dataset.total_samples:
            boards, policies, values, valid_moves = validation_dataset.get_batch(cfg["training"]["batch_size"])
            total_sample_count += len(boards)

            pred_values, pred_policy = model(boards)

            if cfg["training"]["exclude_invalid_moves_from_loss"]:
                pred_policy[~valid_moves] = -1e6

            value_loss = nn.CrossEntropyLoss()(
                pred_values,
                values,
            )
            policy_loss = nn.CrossEntropyLoss()(
                pred_policy,
                policies,
            )
            loss = value_loss + cfg["training"]["policy_loss_weight"] * policy_loss

            results["total_loss"] += loss.item()
            results["value_loss"] += value_loss.item()
            results["policy_loss"] += policy_loss.item()
            results["value_max_correct"] += (pred_values.argmax(dim=1) == values.argmax(dim=1)).sum().item()
            results["policy_max_correct"] += (pred_policy.argmax(dim=1) == policies.argmax(dim=1)).sum().item()

    results["total_loss"] /= total_sample_count
    results["value_loss"] /= total_sample_count
    results["policy_loss"] /= total_sample_count
    results["value_max_correct"] /= total_sample_count
    results["policy_max_correct"] /= total_sample_count

    return results

def run(cfg):
    # Identify the file paths to consider.
    file_paths = []
    for file_path in os.listdir(cfg["training"]["data_read_directory"]):
        if (
            file_path.endswith(".npz") and 
            file_path >= cfg["training"]["minimum_file"] and
            file_path <= cfg["training"]["maximum_file"]
        ):
            file_paths.append(os.path.join(cfg["training"]["data_read_directory"], file_path))

    # Shuffle the files and split them into training and validation sets.
    random.shuffle(file_paths)

    cutoff = int(len(file_paths) * (1 - cfg["training"]["validation_set_size"]))
    training_file_paths = file_paths[:cutoff]
    validation_file_paths = file_paths[cutoff:]

    # Create a game dataset and load the desired files in.
    print("Loading train dataset...")
    train_dataset = GameDataset(cfg["training"]["device"], cfg, shuffle_files=False)
    for file_path in training_file_paths:
        train_dataset.add_file(file_path)
    print("Loaded", train_dataset.total_samples, "samples for training")

    print("Loading validation dataset...")
    validation_dataset = GameDataset(cfg["training"]["device"], cfg, shuffle_files=False)
    for file_path in validation_file_paths:
        validation_dataset.add_file(file_path)
    print("Loaded", validation_dataset.total_samples, "samples for validation")

    try:
        network_config = cfg["networks"][cfg["training"]["network_name"]]

        # Initialize aim run for experiment tracking
        run_obj = aim.Run()
        run_obj["hparams"] = {
            "path": os.getcwd(),
            "network": network_config,
            "config": cfg["training"],
        }
        print("Starting training on run:", run_obj.hash)

        # Start with a randomly initialized model.
        model = NeuralNet(network_config, cfg)
        model.to(cfg["training"]["device"])
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])

        batch_index = 0
        last_validation_evaluation = 0

        for epoch in range(cfg["training"]["num_epochs"]):
            print("Starting epoch", epoch)

            train_dataset.force_shuffle()

            total_sample_count = 0
            while total_sample_count < train_dataset.total_samples:
                boards, policies, values, valid_moves = train_dataset.get_batch(cfg["training"]["batch_size"])

                pred_values, pred_policy = model(boards)

                if cfg["training"]["exclude_invalid_moves_from_loss"]:
                    pred_policy[~valid_moves] = -1e6

                value_loss = nn.CrossEntropyLoss()(
                    pred_values,
                    values,
                )
                policy_loss = nn.CrossEntropyLoss()(
                    pred_policy,
                    policies,
                )
                loss = value_loss + cfg["training"]["policy_loss_weight"] * policy_loss

                loss.backward()

                # Save gradients
                run_obj.track(
                    torch.cat([p.grad.view(-1) for p in model.value_head.parameters()]).norm(),
                    name="value_head_grad_norm",
                    step=batch_index,
                    context={"subset": "train"},
                )
                run_obj.track(
                    torch.cat([p.grad.view(-1) for p in model.policy_head.parameters()]).norm(),
                    name="policy_head_grad_norm",
                    step=batch_index,
                    context={"subset": "train"},
                )

                optimizer.step()
                optimizer.zero_grad()

                training_result = {
                    "total_loss": loss.item() / len(boards),
                    "value_loss": value_loss.item() / len(boards),
                    "policy_loss": policy_loss.item() / len(boards),
                }

                for key, value in training_result.items():
                    run_obj.track(
                        value,
                        name=key,
                        step=batch_index,
                        context={"subset": "train"},
                    )
                
                samples_since_last_validation_evaluation = (batch_index - last_validation_evaluation) * cfg["training"]["batch_size"]
                if samples_since_last_validation_evaluation >= cfg["training"]["num_samples_between_test_evaluations"]:
                    print("Evaluating validation set...")
                    validation_losses = get_validation_losses(model, validation_dataset, cfg)
                    for key, value in validation_losses.items():
                        run_obj.track(
                            value,
                            name=key,
                            step=batch_index,
                            context={"subset": "test"},
                        )
                    last_validation_evaluation = batch_index
                                
                total_sample_count += len(boards)
                batch_index += 1

    finally:
        # Make sure to close the aim run
        run_obj.close()

# def get_gradients_from_random_batch(cfg):
#     file_paths = []
#     for file_path in os.listdir(cfg["training"]["data_read_directory"]):
#         if (
#             file_path.endswith(".npz") and 
#             file_path >= cfg["training"]["minimum_file"] and
#             file_path <= cfg["training"]["maximum_file"]
#         ):
#             file_paths.append(os.path.join(cfg["training"]["data_read_directory"], file_path))

#     # Shuffle the files and split them into training and validation sets.
#     random.shuffle(file_paths)

#     # Load the model
#     model = NeuralNet(cfg["networks"], cfg)
#     model.to(cfg["training"]["device"])
    
#     train_dataset = GameDataset(cfg["training"]["device"], cfg, shuffle_files=False)
#     for file_path in file_paths:
#         train_dataset.add_file(file_path)
#     print("Loaded", train_dataset.total_samples, "samples for training")

#     boards, policies, values, valid_moves = train_dataset.get_batch(cfg["training"]["batch_size"])
    
#     # Forward pass
#     pred_values, pred_policy = model(boards)
    
#     # Get valid moves
#     pred_policy[~valid_moves] = -1e6
    
#     # Calculate losses
#     value_loss = nn.CrossEntropyLoss()(pred_values, values)
#     policy_loss = nn.CrossEntropyLoss()(pred_policy, policies)
#     loss = value_loss + cfg["training"]["policy_loss_weight"] * policy_loss
    
#     # Backward pass
#     loss.backward()

#     gradients = {
#         'grad_norm/value_head': torch.cat([p.grad.view(-1) for p in model.value_head.parameters()]).norm(),
#         'grad_norm/policy_head': torch.cat([p.grad.view(-1) for p in model.policy_head.parameters()]).norm(),
#     }
    
#     print(gradients)
