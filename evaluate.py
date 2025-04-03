"""
Script to evaluate the model on a test set.
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from configs.data_model import ExperimentConfig
from models.mlp_classifier import MLPClassifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="Path to JSON config file.")
    args = parser.parse_args()

    # Create a logger
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Read config files
    with open(args.config, "r", encoding="utf-8") as file_handle:
        config = ExperimentConfig(**json.load(file_handle))

    # Read test dataset
    with open(
        config.dataset_save_path / f"{config.dataset_name.value}.test.pkl", "rb"
    ) as file_handle:
        test_dataset = pickle.load(file_handle)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

    # Read model and model state
    model = MLPClassifier(
        input_dim=config.num_features,
        hidden_dim=config.num_hidden,
        output_dim=config.num_classes,
    )
    model.load_state_dict(
        torch.load(
            config.model_save_path
            / f"{config.name.value}.{config.optimizer_config.name.value}.pth",
            weights_only=True,
        )
    )

    # Evaluate the model on test data
    loss, accuracy = model.evaluate(test_loader, CrossEntropyLoss())

    logger.info(f"Test loss: {loss:.04f}, test accuracy: {accuracy:.04f}.")
