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

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    with open(args.dataset_config, "r", encoding="utf-8") as file_handle:
        dataset_config = DatasetConfig(**json.load(file_handle))

    with open(
        dataset_config.save_path / f"{dataset_config.name}.train.pkl", "rb"
    ) as pickle_handle:
        train_dataset = pickle.load(pickle_handle)

    with open(
        dataset_config.save_path / f"{dataset_config.name}.val.pkl", "rb"
    ) as pickle_handle:
        val_dataset = pickle.load(pickle_handle)

    with open(
        dataset_config.save_path / f"{dataset_config.name}.test.pkl", "rb"
    ) as pickle_handle:
        test_dataset = pickle.load(pickle_handle)

    with open(args.training_config, "r", encoding="utf-8") as file_handle:
        training_config = TrainingConfig(**json.load(file_handle))

    model = MLPClassifier(
        input_dim=dataset_config.num_features,
        hidden_dim=dataset_config.num_hidden,
        output_dim=dataset_config.num_classes,
    )

    # Measure time before training
    start_time = time.time()

    # Training
    if isinstance(training_config.optimizer_config, GradientOptimizerConfig):
        model = train_gradient(
            model,
            train_dataset,
            val_dataset,
            epochs=training_config.epochs,
            learning_rate=training_config.optimizer_config.learning_rate,
            batch_size=training_config.batch_size,
            use_wandb=training_config.use_wandb,
            optimizer=training_config.optimizer_config.name,
            logger=logger,
        )
    elif training_config.optimizer_config.name == "cmaes":
        model = train_cmaes(
            model,
            train_dataset,
            val_dataset,
            epochs=training_config.epochs,
            batch_size=training_config.batch_size,
            use_wandb=training_config.use_wandb,
            logger=logger,
        )
    elif training_config.optimizer_config.name == "layerwise_cmaes":
        model = train_cmaes_layerwise(
            model,
            train_dataset,
            val_dataset,
            epochs=training_config.epochs,
            batch_size=training_config.batch_size,
            use_wandb=training_config.use_wandb,
            logger=logger,
        )
    else:
        raise ValueError(f"Invalid valid training method!")

    # Measure time after training
    elapsed_time = time.time() - start_time

    logger.info(f"Execution Time: {elapsed_time:.2f} seconds")

    torch.save(
        model.state_dict(),
        f"{dataset_config.name}.{training_config.optimizer_config.name.value}.pth",
    )
