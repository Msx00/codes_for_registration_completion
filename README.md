# Liver2 V3
#####

V3-only liver point-cloud completion and registration pipeline.

## Project layout

```text
liver2/
├── data/             dataset loading, augmentation, and collation
├── models/           end-to-end pipeline and GIRNet V3 backbone
├── losses/           registration, correspondence, and physics losses
├── training/         distributed training entry point
├── evaluation/       validation logic and test CLI
└── utils/            shared persistence helpers
completion/           SPAQNet completion component
scripts/              full shell launchers
tests/                focused V3 model tests
```

The short launchers in the repository root forward to `scripts/`, preserving
the original commands. Python entry points use package modules, so imports no
longer depend on the current file layout or ad-hoc `sys.path` changes.

## Train

The launcher resolves the project directory from its own location and defaults
to four GPUs:

```bash
./run_train_multigpu.sh
```

Configuration is supplied with environment variables. For example:

```bash
GPU_IDS=0 WORLD_SIZE=1 \
MAX_TRAIN_SAMPLES=1 MAX_VAL_SAMPLES=1 \
EPOCHS=20 BATCH_SIZE=1 USE_WANDB=0 AMP_DTYPE=fp32 \
./run_train_multigpu.sh
```

Important variables include `DATASET_ROOT`, `COMPLETION_CKPT`, `SAVE_DIR`,
`TRAIN_STAGE`, `REGISTRATION_TARGET_MODE`, `LR`, and the `V3_*` settings.
Additional Python arguments can be appended to the command line.

The equivalent Python entry point is:

```bash
python -m liver2.training.train_multigpu --help
```

## Test

```bash
CHECKPOINT=/path/to/v3/best.pth ./run_test.sh
```

The equivalent Python entry point is:

```bash
python -m liver2.evaluation.test_pipeline --checkpoint /path/to/v3/best.pth
```

Both training and testing reject checkpoints from older GIRNet architectures.




Corresponding paper citation is coming soon...
