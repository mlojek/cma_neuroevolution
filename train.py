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
    CMAOptimizerName,
    ExperimentConfig,
    GradientOptimizerConfig,
)
from models.mlp_classifier import MLPClassifier
from training.train_cmaes import train_cmaes
from training.train_gradient import train_gradient
from training.train_layerwise import train_cmaes_layerwise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="Path to JSON config file.")
    args = parser.parse_args()

    # Setup logger
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Read config
    with open(args.config, "r", encoding="utf-8") as file_handle:
        config = ExperimentConfig(**json.load(file_handle))

    # Read train and val datasets
    with open(
        config.dataset_save_path / f"{config.dataset_name.value}.train.pkl", "rb"
    ) as pickle_handle:
        train_dataset = pickle.load(pickle_handle)

    with open(
        config.dataset_save_path / f"{config.dataset_name.value}.val.pkl", "rb"
    ) as pickle_handle:
        val_dataset = pickle.load(pickle_handle)

    # Create an instance of classfier to train
    model = MLPClassifier(
        input_dim=config.num_features,
        hidden_dim=config.num_hidden,
        output_dim=config.num_classes,
    )

    # Train the model
    start_time = time.time()

    if isinstance(config.optimizer_config, GradientOptimizerConfig):
        model = train_gradient(model, train_dataset, val_dataset, config, logger)
    elif config.optimizer_config.name == CMAOptimizerName.CMAES:
        model = train_cmaes(model, train_dataset, val_dataset, config, logger)
    elif config.optimizer_config.name == CMAOptimizerName.LAYERWISE_CMAES:
        model = train_cmaes_layerwise(model, train_dataset, val_dataset, config, logger)
    else:
        raise ValueError(f"Invalid training method {config.optimizer_config.name}")

    elapsed_time = time.time() - start_time

    logger.info(f"Execution Time: {elapsed_time:.2f} seconds")

    # Save the trained model
    torch.save(
        model.state_dict(),
        config.model_save_path
        / f"{config.dataset_name.value}.{config.optimizer_config.name.value}.pth",
    )
