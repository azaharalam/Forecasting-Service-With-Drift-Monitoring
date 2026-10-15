# Adapted from nanoTabPFN (Pfefferle, Hog, Purucker, Hutter, 2025)
# https://github.com/automl/nanoTabPFN  - Apache License 2.0
# Changes from the original:
#   - removed the breast-cancer evaluation inside the training loop
#   - removed the unused `from sklearn.datasets import *`
#   - train() returns the loss of every step (for plotting)

import random
import time

import h5py
import numpy as np
import schedulefree
import torch
from torch import nn
from torch.utils.data import DataLoader

from model import NanoTabPFNModel


def set_randomness_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_default_device():
    device = "cpu"
    if torch.backends.mps.is_available(): device = "mps"
    if torch.cuda.is_available(): device = "cuda"
    return device


def train(model: NanoTabPFNModel, prior: DataLoader, lr: float = 1e-4,
          device=None, print_every: int = 25):
    """
    Pre-trains the model on synthetic datasets from the prior.
    For each synthetic dataset the model sees X (train+test rows) and y_train,
    and must predict y_test. Loss = cross-entropy on the test rows.

    Returns:
        (model) the trained model
        (list)  loss of every step
    """
    if not device:
        device = get_default_device()
    model.to(device)
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    model.train()
    optimizer.train()

    losses = []
    start = time.time()
    try:
        for step, full_data in enumerate(prior):
            split = full_data["train_test_split_index"]
            data = (full_data["x"].to(device),
                    full_data["y"][:, :split].to(device))
            targets = full_data["y"][:, split:].to(device)

            output = model(data, train_test_split_index=split)
            targets = targets.reshape((-1,)).to(torch.long)
            output = output.view(-1, output.shape[-1])

            loss = criterion(output, targets).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())
            if (step + 1) % print_every == 0:
                recent = np.mean(losses[-print_every:])
                print(f"step {step + 1:5d} | time {time.time() - start:7.1f}s | loss {recent:.4f}")
    except KeyboardInterrupt:
        print("Training interrupted, keeping the model trained so far.")

    # schedule-free optimizer: switch to eval mode so the averaged weights are used
    model.eval()
    optimizer.eval()
    return model, losses


class PriorDumpDataLoader(DataLoader):
    """DataLoader that loads synthetic prior data from an HDF5 dump.

    Args:
        filename (str): Path to the HDF5 file.
        num_steps (int): Number of batches per epoch.
        batch_size (int): Batch size.
        device (torch.device): Device to load tensors onto.
    """

    def __init__(self, filename, num_steps, batch_size, device=None):
        self.filename = filename
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.device = device if device is not None else get_default_device()
        self.pointer = 0
        with h5py.File(self.filename, "r") as f:
            self.max_num_classes = f["max_num_classes"][0]

    def __iter__(self):
        with h5py.File(self.filename, "r") as f:
            for _ in range(self.num_steps):
                end = self.pointer + self.batch_size
                num_features = f["num_features"][self.pointer:end].max()
                num_datapoints_batch = f["num_datapoints"][self.pointer:end]
                max_seq_in_batch = int(num_datapoints_batch.max())
                x = torch.from_numpy(f["X"][self.pointer:end, :max_seq_in_batch, :num_features])
                y = torch.from_numpy(f["y"][self.pointer:end, :max_seq_in_batch])
                train_test_split_index = f["single_eval_pos"][self.pointer:end]

                self.pointer += self.batch_size
                if self.pointer >= f["X"].shape[0]:
                    print("Finished iteration over all stored datasets!")
                    self.pointer = 0

                yield dict(
                    x=x.to(self.device),
                    y=y.to(self.device),
                    train_test_split_index=train_test_split_index[0].item(),
                )

    def __len__(self):
        return self.num_steps
