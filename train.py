"""
Model training script.
"""

import argparse
import json
import logging
import pickle
import time
from pathlib import Path

import torch

from configs.data_model import (
    DatasetConfig,
    GradientOptimizerConfig,
    TrainingConfig,
    CMAOptimizerName
)
from experiments.train_cmaes import train_cmaes
from experiments.train_gradient import train_gradient
from experiments.train_layerwise import train_cmaes_layerwise
from models.mlp_classifier import MLPClassifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_config", type=Path, help="Path to dataset config JSON."
    )
    parser.add_argument(
        "training_config",
        type=Path,
        help="Path to training config JSON.",
    )
    args = parser.parse_args()

    # Setup logger
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Read configs
    with open(args.dataset_config, "r", encoding="utf-8") as file_handle:
        dataset_config = DatasetConfig(**json.load(file_handle))

    with open(args.training_config, "r", encoding="utf-8") as file_handle:
        training_config = TrainingConfig(**json.load(file_handle))

    # Read train and val datasets
    with open(
        dataset_config.save_path / f"{dataset_config.name}.train.pkl", "rb"
    ) as pickle_handle:
        train_dataset = pickle.load(pickle_handle)

    with open(
        dataset_config.save_path / f"{dataset_config.name}.val.pkl", "rb"
    ) as pickle_handle:
        val_dataset = pickle.load(pickle_handle)

    # Create an instance of classfier to train
    model = MLPClassifier(
        input_dim=dataset_config.num_features,
        hidden_dim=dataset_config.num_hidden,
        output_dim=dataset_config.num_classes,
    )

    # Train the model
    start_time = time.time()

    if isinstance(training_config.optimizer_config, GradientOptimizerConfig):
        model = train_gradient(
            model, train_dataset, val_dataset, training_config, logger
        )
    elif training_config.optimizer_config.name == CMAOptimizerName.CMAES:
        model = train_cmaes(model, train_dataset, val_dataset, training_config, logger)
    elif training_config.optimizer_config.name == CMAOptimizerName.LAYERWISE_CMAES:
        model = train_cmaes_layerwise(
            model, train_dataset, val_dataset, training_config, logger
        )
    else:
        raise ValueError(
            f"Invalid training method {training_config.optimizer_config.name}"
        )

    elapsed_time = time.time() - start_time

    logger.info(f"Execution Time: {elapsed_time:.2f} seconds")

    # Save the trained model
    torch.save(
        model.state_dict(),
        training_config.save_path
        / f"{dataset_config.name}.{training_config.optimizer_config.name.value}.pth",
    )
