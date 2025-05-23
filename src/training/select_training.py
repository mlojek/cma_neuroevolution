"""
Function to select the right training function based on the config.
"""

from ..config.data_model import (
    CMAOptimizerName,
    ExperimentConfig,
    GradientOptimizerConfig,
)
from .train_cmaes import train_cmaes
from .train_gradient import train_gradient
from .train_layerwise import train_cmaes_layerwise


def select_training(config: ExperimentConfig) -> callable:
    """
    Depending on experiment config, return the right training function.

    Args:
        config (ExperimentConfig): Experiment configuration object.

    Raises:
        ValueError: When no valid training function was found.

    Returns:
        callable: Training function.
    """
    if isinstance(config.optimizer_config, GradientOptimizerConfig):
        return train_gradient
    if config.optimizer_config.name == CMAOptimizerName.CMAES:
        return train_cmaes
    if config.optimizer_config.name == CMAOptimizerName.LAYERWISE_CMAES:
        return train_cmaes_layerwise

    raise ValueError(f"Invalid training method {config.optimizer_config.name}")
