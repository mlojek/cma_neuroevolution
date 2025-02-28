"""
Data classes and enums that hold data.
"""

from enum import Enum

from pydantic import BaseModel


class OptimizationType(Enum):
    """
    Enum describing the type of optimization strategy.
    """

    GRADIENT = "gradient"
    CMAES = "cmaes"


class GradientMethod(Enum):
    """
    Enum describing gradient optimization method.
    """

    SGD = "sgd"
    ADAM = "adam"


class CMAMethod(Enum):
    """
    Enum describing the type of CMA neuroevolution method.
    """

    WHOLE_MODEL = "whole_model"
    HEAD_FIRST = "head_first"
    HEAD_LAST = "head_last"


class DatasetConfig(BaseModel):
    """
    Configuration related to the dataset.
    """

    batch_size: int
    test_size: float


class LoggingConfig(BaseModel):
    """
    Configuration related to logging the results.
    """

    use_wandb: bool
    log_interval: int


class ExperimentConfig(BaseModel):
    """
    Configuration of an experiment.
    """

    optimization_type: OptimizationType
    optimization_method: GradientMethod | CMAMethod
    epochs: int
    learning_rate: float
    dataset: DatasetConfig
    logging: LoggingConfig
