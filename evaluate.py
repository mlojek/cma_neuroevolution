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

from configs.data_model import DatasetConfig, TrainingConfig
from models.mlp_classifier import MLPClassifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_config", type=Path, help="Path to the dataset configuration JSON file."
    )
    parser.add_argument(
        "training_config",
        type=Path,
        help="Path to the training configuration JSON file.",
    )
    args = parser.parse_args()

    # Create a logger
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Read config files
    with open(args.dataset_config, "r", encoding="utf-8") as file_handle:
        dataset_config = DatasetConfig(**json.load(file_handle))

    with open(args.training_config, "r", encoding="utf-8") as file_handle:
        training_config = TrainingConfig(**json.load(file_handle))

    # Read test dataset
    with open(
        dataset_config.save_path / f"{dataset_config.name.value}.test.pkl", "rb"
    ) as file_handle:
        test_dataset = pickle.load(file_handle)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

    # Read model and model state
    model = MLPClassifier(
        input_dim=dataset_config.num_features,
        hidden_dim=dataset_config.num_hidden,
        output_dim=dataset_config.num_classes,
    )
    model.load_state_dict(
        torch.load(
            training_config.save_path
            / f"{dataset_config.name.value}.{training_config.optimizer_config.name.value}.pth",
            weights_only=True,
        )
    )

    # Evaluate the model on test data
    loss, accuracy = model.evaluate(test_loader, CrossEntropyLoss())

    logger.info(f"Test loss: {loss:.04f}, test accuracy: {accuracy:.04f}.")
