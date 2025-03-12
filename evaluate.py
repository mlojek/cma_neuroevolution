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

from .configs.data_model import DatasetConfig, TrainingConfig
from .models.mlp_classifier import MLPClassifier

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

    # create a logger
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # read the dataset config
    with open(args.dataset_config, "r", encoding="utf-8") as file_handle:
        dataset_config = DatasetConfig(**json.load(file_handle))

    # read the train dataset and make it a dataloader with one batch
    with open(
        dataset_config.save_path / f"{dataset_config.name}.test.pkl", "rb"
    ) as file_handle:
        test_dataset = pickle.load(file_handle)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

    # read the training config
    with open(args.training_config, "r", encoding="utf-8") as file_handle:
        training_config = TrainingConfig(**json.load(file_handle))

    # read the model
    model = MLPClassifier(
        input_dim=dataset_config.num_features,
        hidden_dim=training_config.num_hidden,
        output_dim=dataset_config.num_classes,
    )
    model = model.load_state_dict(
        torch.load(
            training_config.save_path
            / f"{dataset_config.name}.{training_config.optimizer_config.name}.pth",
            weights_only=True,
        )
    )

    # evaluate
    loss_function = CrossEntropyLoss()
    loss, accuracy = model.evaluate(test_loader, loss_function)

    # log results
    logger.info(f"Test loss: {loss}, test accuracy: {accuracy}.")
