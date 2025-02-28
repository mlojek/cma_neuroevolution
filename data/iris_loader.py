"""
Dataloaders for the Iris dataset.
"""

from typing import Tuple

import torch
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from data_model import DatasetConfig


def load_iris_dataset(
    config: DatasetConfig, *, random_seed=42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create Iris dataset torch dataloaders. The samples are split equally for each class.

    Params:
        config (DatasetConfig): Configuration for the dataset.
        random_seed (int): Random seed, default 42.

    Returns:
        Tuple[DataLoader, Dataloader]: Train and test dataloaders.
    """
    # Load the Iris dataset
    iris = datasets.load_iris()
    x, y = iris.data, iris.target

    # Standardize features
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # Convert to PyTorch tensors
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    # Split into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=config.test_size, random_state=random_seed, stratify=y
    )

    # Create DataLoaders
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, test_loader
