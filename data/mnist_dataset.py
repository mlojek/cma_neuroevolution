"""
Loading and prepocessing of the MNIST dataset. The images are flattened into 1D vectors.
"""

# pylint: disable=too-many-locals

from typing import Tuple

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torchvision import datasets as vision_datasets


def load_mnist_dataset(
    train_val_test_ratio: Tuple[float, float, float], *, random_seed=42
) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
    """
    Create MNIST dataset torch dataloaders. The images are flattened into 1D tensors.

    Params:
        train_val_test_ratio (Tuple[float, float, float]): Ratios of train, val and test
            dataset split lengths.
        random_seed (int): Random seed, default 42.

    Returns:
        Tuple[TensorDataset, TensorDataset, TensorDataset]: Train, val and test datasets.
    """
    mnist_train = vision_datasets.MNIST(download=True, root="./data", train=True)

    x, y = torch.flatten(mnist_train.data.float(), start_dim=1), mnist_train.targets

    test_ratio = train_val_test_ratio[2] / sum(train_val_test_ratio)
    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x,
        y,
        test_size=test_ratio,
        random_state=random_seed,
        stratify=y,
    )

    val_ratio = train_val_test_ratio[1] / sum(train_val_test_ratio[:2])
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval,
        y_trainval,
        test_size=val_ratio,
        random_state=random_seed,
        stratify=y_trainval,
    )

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    test_dataset = TensorDataset(x_test, y_test)

    return train_dataset, val_dataset, test_dataset
