"""
Data model of the experiment configuration
"""

from enum import Enum
from typing import Tuple

from pydantic import BaseModel


class DatasetName(Enum):
    """
    Enumeration of dataset available in this project.
    """

    IRIS = "iris"
    "Iris dataset."

    MNIST = "mnist"
    "MNIST dataset."


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

    sigma0: float
    "Starting value for sigma parameter."


class MLPLayers(BaseModel):
    """
    Configuration of MLPClassifier layer sizes.
    """

    input_dim: int
    "Number of input features."

    hidden_dim: int
    "Dimensionality of the hidden layer of the model."

    output_dim: int
    "Number of classes in the classification task."


class EarlyStoppingConfig(BaseModel):
    """
    Configuration of the early stopping.
    """

    patience: int
    "Number of epochs with no improvement in loss value before training is stopped early."

    delta: float
    "Minimum difference in loss value before training is stopped early."


class ExperimentConfig(BaseModel):
    """
    Configuration of the experiment.
    """

    dataset_name: DatasetName
    "Name of the dataset."

    train_val_test_ratios: Tuple[float, float, float]
    "Proportions of the number of training, validation and testing splits."

    batch_size: int
    "Number of samples in one batch."

    epochs: int
    "Number of training epochs."

    use_wandb: bool
    "If true, training stats will be logged to wandb.ai."

    log_interval: int
    "Training stats logging interval in number of epochs."

    random_seed: int
    "Random seed for all randomized processes."

    mlp_layers: MLPLayers
    "Configuration of MLPClassifier layer sizes."

    early_stopping: EarlyStoppingConfig
    "Configuration of the EarlyStopping component."

    optimizer_config: GradientOptimizerConfig | CMAOptimizerConfig
    "Config of either gradient or CMA optimizer."
