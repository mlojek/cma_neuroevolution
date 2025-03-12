"""
Model training script.
"""

import argparse
import json
import logging
import pickle
import time
from pathlib import Path

import psutil
import torch

from configs.data_model import DatasetConfig, TrainingConfig, GradientOptimizerName, GradientOptimizerConfig
from experiments.train_cmaes import train_cmaes
from experiments.train_gradient import train_gradient
from experiments.train_layerwise import train_cmaes_layerwise
from models.mlp_classifier import MLPClassifier

import configs

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

    with open(args.training_config, 'r', encoding='utf-8') as file_handle:
        training_config = TrainingConfig(**json.load(file_handle))

    model = MLPClassifier(
        input_dim=dataset_config.num_features, hidden_dim=training_config.num_hidden, output_dim=dataset_config.num_classes
    )

    # Measure resources before training
    process = psutil.Process()
    cpu_before = process.cpu_percent()
    mem_before = process.memory_info().rss
    start_time = time.time()

    print(type(training_config.optimizer_config))
    print(isinstance(training_config.optimizer_config, GradientOptimizerConfig))

    # Training
    if type(training_config.optimizer_config) == GradientOptimizerConfig:
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

    # Measure resources after training
    cpu_after = process.cpu_percent()
    mem_after = process.memory_info().rss
    elapsed_time = time.time() - start_time

    logger.info(f"CPU Usage: {cpu_after - cpu_before:.2f}%")
    logger.info(f"Memory Usage: {(mem_after - mem_before) / (1024 ** 2):.2f} MB")
    logger.info(f"Execution Time: {elapsed_time:.2f} seconds")

    torch.save(model.state_dict(), f"{dataset_config.name}.{args.method}.pth")
