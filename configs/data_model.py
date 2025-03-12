"""
Data model of the experiment configuration
"""

from enum import Enum
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


class GradientOptimizerName(Enum):
    """
    Enumeration of gradient-based optimizer methods.
    """

    SGD = "sgd"
    "Stochastic Gradient Descent."

    ADAM = "adam"
    "ADAM optimizer."


class GradientOptimizerConfig(BaseModel):
    """
    Configuration of the gradient optimizer.
    """

    name: GradientOptimizerName
    "Name of the gradient-based optmizer."

    learning_rate: float
    "Learning rate of the optimizer."


class CMAOptimizerName(Enum):
    """
    Enumeration of CMA-ES-based optimizer methods.
    """

    CMAES = "cmaes"
    "Whole model CMA-ES."

    LAYERWISE_CMAES = "layerwise_cmaes"
    "Layerwise CMA-ES."


class CMAOptimizerConfig(BaseModel):
    """
    Configuration of the CMA-ES optimizer.
    """

    name: CMAOptimizerName
    "Name of the CMA-ES-based optimizer."

    population_size: int
    "Population size of the CMA-ES algorithm."

    sigma0: int
    "Starting value for sigma parameter."


class TrainingConfig(BaseModel):
    """
    Configuration related to the training of model.
    """

    num_hidden: int
    "Dimensionality of the hidden layer of the model."

    batch_size: int
    "Number of samples in one batch."

    epochs: int
    "Number of training epochs."

    use_wandb: bool
    "If true, training stats will be logged to wandb.ai."

    save_path: Path
    "Path to save the model to."

    optimizer_config: GradientOptimizerConfig | CMAOptimizerConfig
    "Config of either gradient or CMA optimizer."
