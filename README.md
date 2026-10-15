# Forecasting Service With Drift Monitoring

Course project for the fourth-year AI course, University of Toledo.

## Overview

A forecasting service that serves model predictions and monitors incoming data and model performance for drift.

_TODO: add the problem statement, dataset, and goals._

## Team

- _TODO: names_

## Repository Structure

```
data/         Datasets (not tracked in git)
notebooks/    Exploration and experiments
src/          Source code
tests/        Tests
docs/         Reports, proposal, and figures
```

## Setup

_TODO: add environment and installation steps._

## Usage

_TODO: add how to train, serve, and run drift monitoring._

## Status

Initial repository setup. No code yet.

## Midterm prototype: nanoTabPFN end to end

Code in `src/` is based on nanoTabPFN (Pfefferle et al., 2025,
https://github.com/automl/nanoTabPFN, Apache 2.0; see `LICENSE-nanoTabPFN`).

| File | What it does |
|---|---|
| `src/model.py` | Transformer (attention across features and across rows) + sklearn-style classifier. Unmodified from nanoTabPFN. |
| `src/train.py` | Pre-training loop and loader for the synthetic prior data. Lightly adapted. |
| `src/pretrain.py` | Full pre-training; saves `checkpoints/nanotabpfn.pt` and `results/pretrain_loss.png`. |
| `src/demo.py` | Live demo: model size, short live pre-training, prediction on breast cancer. |

### Setup
```
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
mkdir -p data
curl -L -o data/300k_150x5_2.h5 "https://ndownloader.figshare.com/files/58932628?private_link=63fc1ada93e42e388e63"
```

### Run (from the repo root)
```
python src/pretrain.py            # once, before the demo (CPU: roughly 30-60 min)
python src/demo.py                # live demo
python src/demo.py --live-steps 0 # skip the live training part
```
