# AlphaBlokus

Agent for the board game [Blokus](https://en.wikipedia.org/wiki/Blokus), implemented in Rust and trained purely on self-play. As far as I am aware, AlphaBlokus achieves the strongest play of any publicly available Blokus agent.

**[Play in your browser here](https://google.com)**. The browser application runs using WebGPU and WASM, so expect speed to depend dramatically on your local hardware.

todo - insert GIF of gameplay?

## Training methodology

AlphaBlokus is trained from scratch using the classic [AlphaZero](https://arxiv.org/abs/1712.01815) approach, modified to incorporate some ideas from subsequent literature and to suit Blokus.

The neutral network network architecture is as follows:
todo - insert network diagram and details here

## Adaptations
Below are a few of the changes I implemented and/or considered from the classic AlphaZero approach:

#### Vectorized value
Traditional AlphaZero is designed for 1v1 games, where the value of a state can be represented as a single number from -1 to 1. Because Blokus is a four-player game, AlphaBlokus represents the value of a game state as a length-4 vector with elements  which sum to 1, representing the projected game result across all four players. The neural network architecture returns this vector from the value head, and the MCTS stores the full vector in the search tree to inform the search.

#### Fast rollouts
[KataGo](https://arxiv.org/abs/1902.10565), a community effort to reproduce AlphaZero performance on Go, pioneered "fast rollouts". Very briefly, by running most moves in a game with a reduced number of MCTS rollouts, we can increase the number of unique games played to provide a larger volume of unique games as training data for the value head. I found that fast rollouts were effective for increasing training performance, and implemented them in AlphaBlokus.

#### Virtual loss
Virtual loss allows for increased concurrency in the MCTS search by permitting multiple tree searches to take place concurrently. This in turn permits larger batch sizes for more efficient GPU inference. I did not implement virtual loss, because I found it unnecessary for my hardware and architecture. On Vast.ai consumer grade machines, large batch sizes showed little improvement in inference speed over batches of just 128 or 256. Running a significant number of concurrent games produced enough inference demand to keep the GPU saturated with fresh batches from separate games without the addition of virtual loss.

#### Invalid move treatment
Policy loss can be computed in two ways with respect to invalid moves:

1. Compute loss over _all_ moves, thus training the network to predict a probability of 0 for invalid moves.
2. Compute loss over _only valid moves_, and ignore the network's output for invalid moves for the purposes of backpropogation.

In either option, during inference time, only the policy logits associated with valid moves are considered for MCTS search. AlphaBlokus implements Option 2, which produced a very significant improvement (~3x in learning speed) over Option 1 in early training. This was somewhat surprising to me: naively, I expected that minimal network bandwidth would be consumed by learning Blokus game rules given how simple they are, and some online sources indicated both options are comparable.

#### Training on Q
In traditional AlphaZero, the value output of the network is trained to target the final game result (`Z`). Oracle [has proposed](https://medium.com/oracledevs/lessons-from-alphazero-part-4-improving-the-training-target-6efba2e71628) training the network to instead target the MCTS Q value generated from rollouts at the search node. In my experiments, I found that training on a Q target did not produce better performance than training on the final game result, and neither did training on a weighted average of Q and Z. So, AlphaBlokus is trained only on Z.

(Perhaps using fast rollouts reduces the benefit from training on Q, because fast rollouts produces higher game diversity of training data for the value head.)

## System Architecture
AlphaBlokus was trained on Vast.ai-rented machines, for a total cost of <$100 for the final run. There are two types of components, self-play and training:

<img width="326" height="250" alt="Untitled (3)" src="https://github.com/user-attachments/assets/f39647fd-701e-463e-95ac-48f5362ebc7a" />

**Self-play** is implemented as a Rust binary intended to run on a single Vast.ai machine. The binary is responsible for:
- Polling an object store (Cloudflare R2) for newly published models
- Running concurrent games of self-play using the latest available model
- Pushing game data files to the object store for training

Rust is a natural choice for the self-play binary given the significant concurrency requirements involved.

The **training script** is implemented in Python using PyTorch. The training script (in `python/scripts/train_live.py`) polls periodically for new game data files on the object store. When the script finds new data, it trains the network on on `new_samples * sampling_ratio` samples pulled from a window of recent files. Generally, the window size is ~3 million samples, and the sampling ratio is 3.0, indicating that each sample is trained on three times.

In my runs, I found that strong performance was usually reached on Vast with an RTX 3070 machine that has roughly 40 GHz of aggregate CPU capacity. Unlike some other AlphaZero implementations, AlphaBlokus does not share a GPU between multiple machine; each self-play binary runs independently on each instance's GPU and CPU resources.

## Inference
Efficient GPU inference is critical to scaling AlphaZero-style training at reasonable cost. For self-play on Vast, inference is done using the TensorRT engine running models stored in the ONNX format.

The AlphaBlokus inference implementation uses three CUDA streams to achieve strong GPU utilization. One stream is responsible for copying data from the CPU to the GPU, one stream is responsible for running inference on the GPU, and one stream is responsible for copying inference results back from the GPU to the CPU. These streams are synchronized with one another to allow for overlapping of data transfer and inference without sacrificing correctness. This implementation is written in C++ with a Rust FFI bridge, and is available in `src/tensorrt/cpp/tensorrt.cpp` and `src/inference/tensorrt/`.

<img width="663" height="377" alt="Untitled (2)" src="https://github.com/user-attachments/assets/827555bb-bdb7-4997-ad3c-3dffb212ecd1" />

For non-NVIDIA systems, inference through ORT is available, but is not optimized and should not be used for heavy self-play.

## Gameplay
As far as I'm aware, the state of the art computer opponent for Blokus has been [Pentobi](https://github.com/enz/pentobi), so I used it as the primary benchmark for AlphaBlokus skill.

The strongest AlphaBlokus model, running with 2000 rollouts, achieves a win rate of TODO% against Pentobi head-to-head at maximum difficulty. Importantly, I have not attempted to rigorously compare win rates using equal compute; Pentobi is CPU-only while AlphaBlokus performs best using a GPU for inference. However, even when I run AlphaBlokus inference on CPU on my local machine, it plays each turn faster than Pentobi at maximum difficulty. So, per unit time, AlphaBlokus is a stronger player out of the box.

I don't know if AlphaBlokus is superhuman. It's certainly better than me, but I'm not very good at Blokus. If you're a strong Blokus player, I'd love to hear your perspective.

## Install / Contributing
TODO - make it possible to install and run locally+on vast, even if not train? Provide a claude script to install.
