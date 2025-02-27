"""
Dataloaders for the Iris dataset.
"""

from typing import Tuple

import torch
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


def load_iris_dataset(
    batch_size=16, test_size=0.2, seed=42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create Iris dataset torch dataloaders. The samples are split equally for each class.

    Params:
        batch_size (int): Number of samples in batch.
        test_size (float): The ratio of samples to put in the train split.
        seed (int): Random seed.

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
        x, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Create DataLoaders
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
