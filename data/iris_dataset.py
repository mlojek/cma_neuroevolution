"""
Dataloaders for the Iris dataset.
"""

# pylint: disable=too-many-locals

from typing import Tuple

import torch
from sklearn import datasets
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset


def load_iris_dataset(
    train_val_test_ratio: Tuple[float, float, float], *, random_seed=42
) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
    """
    Create Iris dataset torch datasets. The samples are split equally for each class.

    Params:
        train_val_test_ratio (Tuple[float, float, float]): Ratios of train, val and train
            dataset split lengths.
        random_seed (int): Random seed, default 42.

    Returns:
        Tuple[TensorDataset, TensorDataset, TensorDataset]: Train, val and test datasets.
    """
    iris = datasets.load_iris()
    x, y = iris.data, iris.target  # pylint: disable=no-member

    x_all = torch.tensor(x, dtype=torch.float32)
    y_all = torch.tensor(y, dtype=torch.long)

    test_ratio = train_val_test_ratio[2] / sum(train_val_test_ratio)
    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x_all, y_all, test_size=test_ratio, random_state=random_seed, stratify=y_all
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
