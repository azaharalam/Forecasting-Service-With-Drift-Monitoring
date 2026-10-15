"""
Live end-to-end demo of nanoTabPFN.

    python src/demo.py
    python src/demo.py --live-steps 0     # skip the live training part

Part 1  model size
Part 2  short live pre-training from scratch (only to show the loss going down)
Part 3  load the fully pre-trained model and predict on a real dataset
        (breast cancer) in one forward pass - no training on this dataset
"""

import argparse
import time

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from model import NanoTabPFNClassifier, NanoTabPFNModel
from train import PriorDumpDataLoader, get_default_device, set_randomness_seed, train


def header(text):
    print("\n" + "=" * 60 + f"\n{text}\n" + "=" * 60)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/300k_150x5_2.h5")
    p.add_argument("--checkpoint", default="checkpoints/nanotabpfn.pt")
    p.add_argument("--live-steps", type=int, default=50)
    args = p.parse_args()

    set_randomness_seed(0)
    device = get_default_device()
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt["config"]

    header("Part 1: the model")
    model = NanoTabPFNModel(**config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Transformer with {config['num_layers']} layers, embedding size "
          f"{config['embedding_size']}, {config['num_attention_heads']} heads")
    print(f"Parameters: {n_params:,} | device: {device}")

    if args.live_steps > 0:
        header(f"Part 2: live pre-training on synthetic data ({args.live_steps} steps)")
        prior = PriorDumpDataLoader(args.data, num_steps=args.live_steps,
                                    batch_size=32, device=device)
        train(model, prior, lr=4e-3, device=device, print_every=10)
        print("(this short run is only to show the method; Part 3 uses the full model)")

    header("Part 3: prediction on a real dataset (breast cancer)")
    print(f"Loaded pre-trained model: {ckpt['steps']} steps, "
          f"{ckpt['train_minutes']:.1f} min of training")
    model = NanoTabPFNModel(**config)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    clf = NanoTabPFNClassifier(model, device)

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=0, stratify=y)
    print(f"Train rows: {len(X_train)} | test rows: {len(X_test)} | features: {X.shape[1]}")

    start = time.time()
    clf.fit(X_train, y_train)           # only stores the training rows (context)
    prob = clf.predict_proba(X_test)    # one forward pass, no weight updates
    seconds = time.time() - start
    pred = prob.argmax(axis=1)

    print(f"\nROC AUC : {roc_auc_score(y_test, prob[:, 1]):.4f}")
    print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
    print(f"Fit + predict time: {seconds:.2f} s")


if __name__ == "__main__":
    main()
