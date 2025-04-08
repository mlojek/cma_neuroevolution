"""
Function to load the right dataset based on given experiment config.
"""

from typing import Tuple

from torch.utils.data import TensorDataset

from configs.data_model import DatasetName, ExperimentConfig
from data.iris_dataset import load_iris_dataset
from data.mnist_dataset import load_mnist_dataset


def load_dataset(
    config: ExperimentConfig,
) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
    """
    Load dataset based on given experiment config.

    Args:
        config (ExperimentConfig): Experiment configuration.

    Raises:
        ValueError: When no dataset was found.

    Returns:
        Tuple[TensorDataset. TensorDataset, TensorDataset]: Train, val and test datasets.
    """
    # Read train and val datasets
    match config.dataset_name:
        case DatasetName.IRIS:
            return load_iris_dataset(config.train_val_test_ratios)
        case DatasetName.MNIST:
            return load_mnist_dataset(config.train_val_test_ratios)
        case _:
            raise ValueError(f"Invalid dataset name {config.dataset_name.name}!")
