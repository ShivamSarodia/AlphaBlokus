# Python Utilities

This directory contains supplemental Python logic for working with AlphaBlokus.

Most critically, scripts/train_live.py is used for training concurrently with self-play in Rust.

## Elo Workflow

One workflow implemented here is the ability to compute Elo ratings.

### Scripts

- `python/scripts/generate_quad_arena_batch.py`
  Reads a manifest TOML and generates a batch of `QuadArena` configs.
- `python/scripts/run_quad_arena_batch.py`
  Runs a generated config directory sequentially via `cargo run --release --bin self-play`.
- `python/scripts/compute_quad_arena_elo.py`
  Reads the per-config JSONL result files and computes offline Elo plus confidence intervals.

### Configurations

See configs/self_play/elos/ for example configurations.

### Usage

Generate configs:

```bash
python3 python/scripts/generate_quad_arena_batch.py \
  --manifest configs/self_play/elos/full_v3_milestones_elo_manifest.toml
```

This writes configs into:

- `configs/self_play/elos/full_v3_milestones_elo`

To run the batch:

```bash
python3 python/scripts/run_quad_arena_batch.py \
  --config-directory configs/self_play/arenas/full_v3_milestones_elo \
  --resume
```

`--resume` skips configs whose JSONL result file already exists and is non-empty.

To compute elo once the batch is complete:

```bash
python3 python/scripts/compute_quad_arena_elo.py \
  --results-directory /tmp/alphablokus_quad_arena_results/full_v3_milestones_elo
```
