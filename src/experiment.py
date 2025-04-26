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

from .config.data_model import ExperimentConfig
from .data.load_dataset import load_dataset
from .models.mlp_classifier import MLPClassifier
from .training.select_training import select_training

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
        "--runs", type=int, default=1, help="Number of model trainings to perform."
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

    # Read dataset
    train_dataset, val_dataset, test_dataset = load_dataset(config)

    # dictionary to store statistics
    dataframe_rows = []

    # choose the right training function depending on configuration
    train_function = select_training(config)

    # Train the model
    for run_num in range(args.runs):
        logger.info(f"Starting run {run_num+1}/{args.runs}")

        # Create an instance of classfier to train
        model = MLPClassifier(
            **config.mlp_layers.model_dump(), random_seed=config.random_seed
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
        f"{config.dataset_name.value}.{config.optimizer_config.name.value}.csv"
    )
