# Habitat Classification from Ground-Level Imagery Using Deep Neural Networks

This repository contains the core code needed to reproduce the main training experiments from the paper ["Habitat Classification from Ground-Level Imagery Using Deep Neural Networks"](https://arxiv.org/abs/2507.04017).

In the paper, the models are evaluated for habitat classification from ground-level images collected in the UK Countryside Survey. This release keeps only the code required to reproduce the main CS experiments.

The public version is intentionally minimal. It focuses on reproducing:

- standard supervised training with CNN backbones
- standard supervised training with Swin Transformers
- supervised contrastive pretraining on the CS dataset
- linear evaluation after supervised contrastive pretraining

The image data are not included in this repository. **Please contact the authors to request access to the data.**

## Repository layout

```text
.
├── config/
│   ├── cs_cnn.yaml
│   ├── cs_swint.yaml
│   ├── cs_supcon_pretrain.yaml
│   ├── cs_supcon_linear.yaml
│   └── config_parser.py
├── data/
├── methods/
├── models/
├── utils/
├── main.py
└── main_supcon.py
```

## Environment setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you need a CUDA-specific PyTorch build, install `torch` and `torchvision` according to the official PyTorch instructions for your platform before installing the remaining requirements.

The training scripts log to Weights & Biases by default. If you do not want online logging, run in offline mode:

```bash
export WANDB_MODE=offline
```

## Data layout

This code expects the CS dataset to be arranged as separate train and test folders:

```text
data/
├── CS_Xplots_2019_2023_train/
│   ├── CS_Xplots_2019_23_NEW02OCT24.csv
│   └── *.jpg
└── CS_Xplots_2019_2023_test/
    ├── CS_Xplots_2019_23_NEW02OCT24.csv
    └── *.jpg
```

Important details:

- The dataset folder used in the config must end in `_train`.
- The test loader is derived automatically by replacing `_train` with `_test`.
- The index CSV is resolved relative to each image folder, so the CSV file must be present inside both the train and test folders. **Please contact the authors to request the index CSV.**

The loader reads the CSV columns used in `data/dataset.py`. At minimum, the metadata table should contain:

- `file`
- `ID`
- `BH_PLOT_DESC`

`BH_POLYDESC` is also used when resolving boundary labels.

## Configs

Four clean public configs are provided:

- [`config/cs_cnn.yaml`](config/cs_cnn.yaml): standard supervised CNN training
- [`config/cs_swint.yaml`](config/cs_swint.yaml): standard supervised Swin Transformer training
- [`config/cs_supcon_pretrain.yaml`](config/cs_supcon_pretrain.yaml): supervised contrastive pretraining
- [`config/cs_supcon_linear.yaml`](config/cs_supcon_linear.yaml): linear evaluation after supervised contrastive pretraining

These configs are intended as reproducible starting points, not as a full sweep or ablation framework.

## Reproducing the experiments

### 1. Train a CNN baseline

```bash
python main.py --config config/cs_cnn.yaml --run-id cs_cnn_run
```

### 2. Train a Swin Transformer baseline

```bash
python main.py --config config/cs_swint.yaml --run-id cs_swint_run
```

### 3. Run supervised contrastive pretraining

```bash
python main_supcon.py --config config/cs_supcon_pretrain.yaml --run-id cs_supcon_pretrain
```

This stage saves encoder checkpoints to:

```text
./experiments/cs_supcon_pretrain/checkpoints/
```

### 4. Run linear evaluation after SupCon pretraining

First make sure `training.supcon_conf.prt_dir` and `training.supcon_conf.prt_filename` in [`config/cs_supcon_linear.yaml`](config/cs_supcon_linear.yaml) point to a checkpoint produced by the pretraining stage.

Then run:

```bash
python main_supcon.py --config config/cs_supcon_linear.yaml --run-id cs_supcon_linear
```

## Outputs

Each run writes to:

```text
<save_dir>/<run-id>/
├── checkpoints/
└── results/
```

Typical outputs include:

- saved model checkpoints
- confusion matrices
- CSV files for correctly and incorrectly classified samples

## Train/test split utility

[`data/testset_generator.py`](data/testset_generator.py) is included as the original utility for generating a grouped stratified test split from a full image folder.

Note that it currently moves files rather than copying them, so review the paths carefully before using it.

## Scope

This public repository excludes notebooks, visualisation utilities, sweep scripts, experiment logs, and image assets. The goal is to keep the release focused on the core code needed to reproduce the main CS experiments reported in the paper.

## Citation

If you use this repository, please cite the paper:

```text
Hongrui Shi, Lisa Norton, Lucy Ridding, Simon Rolph, Tom August,
Claire M. Wood, Lan Qie, Petra Bosilj, James M. Brown.
Habitat Classification from Ground-Level Imagery Using Deep Neural Networks.
arXiv:2507.04017, 2025.
```
**Contact: Dr Hongrui Shi (`hshi@lincoln.ac.uk`)**
