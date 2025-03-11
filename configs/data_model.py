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


class ModelConfig(BaseModel):
    """
    Configuration related to the MLPClassifier model.
    """

    input_dim: int
    "Number of input features to the model."

    hidden_dim: int
    "Number of neurons in the hidden layer."

    output_dim: int
    "Number of classes in the classification task."


class TrainingConfig(BaseModel):
    # TODO docstrings
    # TODO class members
    batch_size: int
