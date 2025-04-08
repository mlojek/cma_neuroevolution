"""
Train and evaluate model to gather data about it's performance.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
from torch import nn
from torch.utils.data import DataLoader

from configs.data_model import (
    CMAOptimizerName,
    DatasetName,
    ExperimentConfig,
    GradientOptimizerConfig,
)
from data.iris_dataset import load_iris_dataset
from data.mnist_dataset import load_mnist_dataset
from models.mlp_classifier import MLPClassifier
from training.train_cmaes import train_cmaes
from training.train_gradient import train_gradient
from training.train_layerwise import train_cmaes_layerwise
from training.select_training import select_training

DATAFRAME_COLUMNS = [
    "time",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "test_loss",
    "test_acc",
    "model_evals",
    "grad_evals",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="Path to JSON config file.")
    parser.add_argument(
        "--runs", type=int, default=25, help="Number of model trainings to perform."
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

    # Read train and val datasets
    match config.dataset_name:
        case DatasetName.IRIS:
            train_dataset, val_dataset, test_dataset = load_iris_dataset(
                config.train_val_test_ratios
            )
        case DatasetName.MNIST:
            train_dataset, val_dataset, test_dataset = load_mnist_dataset(
                config.train_val_test_ratios
            )
        case _:
            raise ValueError(f"Invalid dataset name {config.dataset_name.name}!")

    # dictionary to store statistics
    dataframe_rows = []

    # choose the right training function depending on configuration
    train_function = select_training(config)

    # Train the model
    for run_num in range(args.runs):
        logger.info(f"Starting run {run_num+1}/{args.runs}")

        # Create an instance of classfier to train
        model = MLPClassifier(
            input_dim=config.num_features,
            hidden_dim=config.num_hidden,
            output_dim=config.num_classes,
        )

        row = []

        start_time = time.time()
        model = train_function(model, train_dataset, val_dataset, config, logger)
        row = [time.time() - start_time]

        loss_fn = nn.CrossEntropyLoss()

        row.extend(
            [
                *model.evaluate(DataLoader(train_dataset), loss_fn),
                *model.evaluate(DataLoader(val_dataset), loss_fn),
                *model.evaluate(DataLoader(test_dataset), loss_fn),
                model.eval_counter,
                model.grad_counter,
            ]
        )

        dataframe_rows.append(row)

    experiment_stats = pd.DataFrame(dataframe_rows, columns=DATAFRAME_COLUMNS)

    print(experiment_stats.describe())

    experiment_stats.to_csv(
        config.model_save_path
        / f"{config.dataset_name.value}.{config.optimizer_config.name.value}.csv"
    )
