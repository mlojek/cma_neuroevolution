"""
Find the optimal hyperparameter values that minimize test set loss using optuna.
"""

import argparse
import json
import logging
from functools import partial
from logging import Logger
from pathlib import Path

import optuna
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from configs.data_model import (
    CMAOptimizerConfig,
    ExperimentConfig,
    GradientOptimizerConfig,
)
from data.load_dataset import load_dataset
from models.mlp_classifier import MLPClassifier
from training.select_training import select_training


def tuning_objective(config: ExperimentConfig, logger: Logger, trial) -> float:
    """
    Optuna objective to tune optimizer hyperparameters. The objective is to minimize crossentropy
    loss on the test split of dataset.

    Args:
        config (ExperimentConfig): Experiment configuration.
        logger (Logger): Logger for the training function.
        trial: Optuna trial parameter.

    Returns:
        float: Model's crossentropy loss on test set.
    """
    if isinstance(config.optimizer_config, GradientOptimizerConfig):
        config.optimizer_config.learning_rate = trial.suggest_float(
            "learning_rate", 0, 10
        )
    elif isinstance(config.optimizer_config, CMAOptimizerConfig):
        config.optimizer_config.population_size = trial.suggest_int(
            "population_size", 2, 30
        )
        config.optimizer_config.sigma0 = trial.suggest_float("sigma", 0, 10)

    train_dataset, val_dataset, test_datset = load_dataset(config)
    training_function = select_training(config)
    model = MLPClassifier(config.num_features, config.num_hidden, config.num_classes)

    trained_model = training_function(model, train_dataset, val_dataset, config, logger)

    return trained_model.evaluate(DataLoader(test_datset), CrossEntropyLoss())[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", type=Path, help="Path to experiment config JSON file."
    )
    args = parser.parse_args()

    # Setup logger
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Read config
    with open(args.config, "r", encoding="utf-8") as file_handle:
        config = ExperimentConfig(**json.load(file_handle))

    # Perform tuning
    study = optuna.create_study(direction="minimize")
    objective = partial(tuning_objective, config, logger)
    study.optimize(objective, n_trials=100)
    best_trial = study.best_trial

    logger.info(f"Optuna found best values: {best_trial.params}.")
    logger.info(f"Lowest found loss value of {best_trial.value:.4f}.")
