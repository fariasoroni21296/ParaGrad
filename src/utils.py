import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import pandas as pd
import os

def load_mnist(data_path="../data/raw"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST(
        root=data_path,
        train=True,
        download=True,
        transform=transform
    )

    return dataset

def split_dataset(dataset, num_splits):
    length = len(dataset)
    split_size = length // num_splits

    subsets = []

    for i in range(num_splits):
        start = i * split_size

        if i == num_splits - 1:
            end = length
        else:
            end = (i + 1) * split_size

        subsets.append(Subset(dataset, range(start, end)))

    return subsets

def calculate_speedup(seq_time, par_time):
    return seq_time / par_time

def calculate_efficiency(speedup, num_processes):
    return speedup / num_processes

def save_metrics(metrics, path="../result/metrics/results.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    df = pd.DataFrame([metrics])

    if os.path.exists(path):
        df.to_csv(path, mode='a', header=False, index=False)
    else:
        df.to_csv(path, index=False)

def plot_results(labels, values, title, ylabel, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.savefig(save_path)
    plt.close()