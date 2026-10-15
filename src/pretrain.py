"""
Step 1-2 of the method: pre-train nanoTabPFN on synthetic tabular datasets.

Run once (e.g. overnight on CPU):
    python src/pretrain.py
    python src/pretrain.py --steps 500        # shorter run

Saves:
    checkpoints/nanotabpfn.pt     trained model + its configuration
    results/pretrain_loss.png     loss curve
"""

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from model import NanoTabPFNModel
from train import PriorDumpDataLoader, get_default_device, set_randomness_seed, train

# same configuration as the nanoTabPFN paper
MODEL_CONFIG = dict(embedding_size=96, num_attention_heads=4,
                    mlp_hidden_size=192, num_layers=3, num_outputs=2)


def plot_loss(losses, path):
    window = max(1, len(losses) // 50)
    smooth = np.convolve(losses, np.ones(window) / window, mode="valid")
    plt.figure(figsize=(6, 4))
    plt.plot(losses, alpha=0.3, label="loss per step")
    plt.plot(np.arange(window - 1, len(losses)), smooth, label=f"moving average ({window})")
    plt.xlabel("training step")
    plt.ylabel("cross-entropy loss")
    plt.title("nanoTabPFN pre-training on synthetic data")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/300k_150x5_2.h5")
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=4e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="checkpoints/nanotabpfn.pt")
    args = p.parse_args()

    set_randomness_seed(args.seed)
    device = get_default_device()
    model = NanoTabPFNModel(**MODEL_CONFIG)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device} | parameters: {n_params:,} | steps: {args.steps} | batch: {args.batch_size}")

    prior = PriorDumpDataLoader(args.data, num_steps=args.steps,
                                batch_size=args.batch_size, device=device)
    start = time.time()
    model, losses = train(model, prior, lr=args.lr, device=device)
    minutes = (time.time() - start) / 60

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": MODEL_CONFIG, "state_dict": model.state_dict(),
                "losses": losses, "steps": len(losses), "train_minutes": minutes}, out)

    Path("results").mkdir(exist_ok=True)
    plot_loss(losses, "results/pretrain_loss.png")
    print(f"\nDone in {minutes:.1f} min. Saved {out} and results/pretrain_loss.png")


if __name__ == "__main__":
    main()
