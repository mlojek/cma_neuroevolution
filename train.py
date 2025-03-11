"""
Model training script.
"""

import argparse
import json
import logging
import pickle
import time
import psutil
from pathlib import Path

import torch

from configs.data_model import DatasetConfig
from experiments.train_cmaes import train_cmaes
from experiments.train_gradient import train_gradient
from models.mlp_classifier import MLPClassifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "method", choices=["adam", "sgd", "cmaes"], help="Optimization type."
    )
    parser.add_argument(
        "dataset_config", type=Path, help="Path to dataset config JSON."
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="If specified, loss and accuracy data will be logged to wandb.ai.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    with open(args.dataset_config, "r", encoding="utf-8") as file_handle:
        dataset_config = DatasetConfig(**json.load(file_handle))

    with open(dataset_config.save_path / f"{dataset_config.name}.train.pkl", "rb") as pickle_handle:
        train_dataset = pickle.load(pickle_handle)

    with open(dataset_config.save_path / f"{dataset_config.name}.val.pkl", "rb") as pickle_handle:
        val_dataset = pickle.load(pickle_handle)

    with open(dataset_config.save_path / f"{dataset_config.name}.test.pkl", "rb") as pickle_handle:
        test_dataset = pickle.load(pickle_handle)

    model = MLPClassifier(
        input_dim=dataset_config.num_features, output_dim=dataset_config.num_classes
    )

    # Measure resources before training
    process = psutil.Process()
    cpu_before = process.cpu_percent()
    mem_before = process.memory_info().rss
    start_time = time.time()

    # Training
    if args.method in ["adam", "sgd"]:
        model = train_gradient(
            model,
            train_dataset,
            val_dataset,
            epochs=10,
            learning_rate=0.01,
            batch_size=16,
            use_wandb=args.use_wandb,
            optimizer=args.method,
            logger=logger,
        )
    elif args.method == "cmaes":
        model = train_cmaes(
            model,
            train_dataset,
            val_dataset,
            epochs=10,
            batch_size=16,
            use_wandb=args.use_wandb,
            logger=logger,
        )
    else:
        raise ValueError(f"{args.method} is not a valid training method!")

    # Measure resources after training
    cpu_after = process.cpu_percent()
    mem_after = process.memory_info().rss
    elapsed_time = time.time() - start_time

    logger.info(f"CPU Usage: {cpu_after - cpu_before:.2f}%")
    logger.info(f"Memory Usage: {(mem_after - mem_before) / (1024 ** 2):.2f} MB")
    logger.info(f"Execution Time: {elapsed_time:.2f} seconds")

    print(model.eval_counter)

    torch.save(model.state_dict(), f"iris.{args.method}.pth")
