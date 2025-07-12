from omegaconf import OmegaConf

from alpha_blokus import benchmark_nn

def run(cfg):
    OmegaConf.set_struct(cfg, False)
    default_benchmark_cfg = OmegaConf.merge(
        cfg,
        {
            "benchmark_duration": 5,
            "num_warmup_batches": 10,
        }
    )
    OmegaConf.set_struct(default_benchmark_cfg, True)

    results = {}
    for batch_size in [64, 128, 256, 512]:
        print("Starting benchmark for batch size: ", batch_size)
        default_benchmark_cfg.networks.main.batch_size = batch_size
        results[batch_size] = benchmark_nn.run(default_benchmark_cfg, verbose=False)

    for k in results:
        print(k)
        print(results[k])
        print()

