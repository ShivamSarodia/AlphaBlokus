# AlphaBlokus

Agent for the board game [Blokus](https://en.wikipedia.org/wiki/Blokus), implemented in Rust and trained purely on self-play. As far as I am aware, AlphaBlokus achieves the strongest play of any publicly available Blokus agent.

todo - insert GIF of gameplay?

---

# Play in browser

Play against the agent here: __ (todo). The application runs using WebGPU and WASM, so do expect speed to depend dramatically on your local hardware.

# Training methodology

AlphaBlokus is trained from scratch using the classic [AlphaZero](https://arxiv.org/abs/1712.01815) approach, modified to incorporate some ideas from subsequent literature and to suit Blokus.

## Network

todo - insert network diagram and details here

## Modifications / Considerations
^ todo better title?

Below are a few changes implemented and/or considered from the classic AlphaZero approach:

### Vectorized value
Traditional AlphaZero is designed for 1v1 games, where the value of a state can be represented as a single number from -1 to 1. Because Blokus is a four-player game, AlphaBlokus represents the value of a game state as a length-4 vector with elements  which sum to 1, representing the projected game result across all four players. The neural network architecture returns this vector from the value head, and the MCTS stores the full vector in the search tree to inform the search.

### Fast rollouts
KataGo, a community effort to reproduce AlphaZero performance on Go, pioneered "fast rollouts". Very briefly, by running most moves in a game with a smaller number of MCTS rollouts, we increase the number of unique games played to provide more helpful training data for the value head. I found that fast rollouts were effective for increasing training performance, and implemented them in AlphaBlokus.

### Virtual loss
Virtual loss allows for increased concurrency in the MCTS search by permitting multiple tree searches to take place concurrently. This in turn permits larger batch sizes for more efficient GPU inference. I did not implement virtual loss, because I found it unnecessary for my hardware. On Vast.ai consumer grade machines, large batch sizes showed little improvement in inference speed over batches of just 128 or 256. Running a significant number of concurrent games produced enough inference demand to keep the GPU saturated with fresh batches from separate games without the addition of virtual loss.

### Invalid move treatment
Policy loss can be computed in two ways with respect to invalid moves:

1. Compute loss over _all_ moves, thus training the network to predict a probability of 0 for invalid moves.
2. Compute loss over _only valid moves_, and ignore the network's output for invalid moves for the purposes of backpropogation.

In either option, during inference time, only the policy logits associated with valid moves are considered for MCTS search. AlphaBlokus implements Option 2, which produced a very significant improvement (~3x in learning speed) over Option 1 in early training. This was somewhat surprising to me. Naively, I expected that minimal network bandwidth would be consumed by learning Blokus game rules given how simple they are, and some sources indicated both options are comparable.

### Training on Q
In traditional AlphaZero, the value output of the network is trained to target the final game result. Oracle [has proposed](https://medium.com/oracledevs/lessons-from-alphazero-part-4-improving-the-training-target-6efba2e71628) training the network to instead target the MCTS Q value generated from rollouts at the search node. In my experiments, I found that training on a Q target did not perform better than training on the standard final game result, so AlphaBlokus was trained only on final game results.

# System Architecture
todo - insert image

AlphaBlokus was trained on Vast.ai-rented machines. There's two components, self-play and training.

*Self-play* is implemented as a Rust binary intended to run on a single Vast.ai machine. The binary is responsible for:
- Polling S3 for newly published models in a models/ directory
- Running concurrent games of self-play using the latest model
- Writing game data files to S3 for training

The *training script* is implemented in Python using PyTorch. The training script (in python/scripts/train_live.py) polls periodically for new game data files on S3. When it finds new data, it trains the network on on `new_samples * sampling_ratio` samples pulled from a window of recent files. Generally, the window size is ~3 million samples, and the sampling ratio is 3.0, indicating that each sample is trained on three times.

In my runs, I found that strong performance was usually reached on Vast with an RTX 3070 machine that has at least 40 GHz of CPU. Unlike some other AlphaZero implementations, AlphaBlokus does not share a GPU between multiple instances, and each self-play binary runs independently on each instance's GPU and CPU resources.

# Inference
Efficient GPU inference is critical to scaling AlphaZero-style training at reasonable cost. For self-play on Vast, inference is done using TensorRT on models stored in ONNX format.

todo - insert diagram

The AlphaBlokus inference implementation uses three CUDA streams to achieve strong GPU utilization. One stream is responsible for copying data from the CPU to the GPU, one stream is responsible for running inference on the GPU, and one stream is responsible for copying inference results back from the GPU to the CPU. These streams are synchronized with one another to allow for overlapping of data transfer and inference without sacrificing correctness. This implementation is written in C++ with a Rust FFI bridge, and is available in src/tensorrt/cpp/tensorrt.cpp and src/inference/tensorrt/.

For non-NVIDIA systems, inference through ORT is available, but is not optimized and should not be used for heavy self-play.

# Gameplay Skill
As far as I'm aware, the state of the art computer opponent for Blokus has been [Pentobi](https://github.com/enz/pentobi), so I used it as a benchmark for AlphaBlokus skill.

The strongest AlphaBlokus model running with 2000 rollouts, achieves a win rate of TODO% against Pentobi head-to-head at level 9, the maximum Pentobi level. todo more details? Importantly, I have not attempted to compare win rates with equal compute because Pentobi is primarily bottlenecked on CPU while AlphaBlokus is bottlenecked on GPU, but even with Pentobi running its neural network evaluation on CPU it is faster per move than Pentobi on my device.

I also don't know if AlphaBlokus is superhuman. It's certainly better than me, but I'm not very good at Blokus. If you're a strong Blokus player, I'd love to hear your perspective.

# Install / Contributing
TODO - make it possible to install and run locally+on vast, even if not train? Provide a claude script to install.
