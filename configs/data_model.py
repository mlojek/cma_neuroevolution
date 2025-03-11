"""
Data model of the experiment configuration
"""

from pathlib import Path
from typing import Tuple

from pydantic import BaseModel


class DatasetConfig(BaseModel):
    """
    Configuration related to the dataset used.
    """

    name: str
    "Name of the dataset."

    train_val_test_ratios: Tuple[float, float, float]
    "Proportions of the number of training, validation and testing splits."

    save_path: Path
    "Directory path to save the dataset pickles."

    num_features: int
    "Number of input features."

    num_classes: int
    "Number of classes in the classification task."


class TrainingConfig(BaseModel):
    # TODO docstrings
    # TODO class members
    batch_size: int
    num_hidden: int
