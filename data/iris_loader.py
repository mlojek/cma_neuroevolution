"""
Dataloaders for the Iris dataset.
"""

from typing import Tuple

import torch
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset

from data_model import DatasetConfig


def load_iris_dataset(
    train_val_test_ratio: Tuple[float, float, float], *, random_seed=42
) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
    """
    Create Iris dataset torch datasets. The samples are split equally for each class.

    Params:
        train_val_test_ratio (Tuple[float, float, float]): TODO
        random_seed (int): Random seed, default 42.

    Returns:
        Tuple[TensorDataset, TensorDataset, TensorDataset]: Train, val and test datasets.
    """
    # Load the Iris dataset
    iris = datasets.load_iris()
    x, y = iris.data, iris.target

    # Standardize features
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # Convert to PyTorch tensors
    x_all = torch.tensor(x, dtype=torch.float32)
    y_all = torch.tensor(y, dtype=torch.long)

    # Split into train and test sets
    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x_all, y_all, test_size=config.test_size, random_state=random_seed, stratify=y
    )

    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval,
        y_trainval,
        test_size=config.test_size,
        random_state=random_seed,
        stratify=y,
    )

    # Create DataLoaders
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    test_dataset = TensorDataset(x_test, y_test)

    return train_dataset, val_dataset, test_dataset
