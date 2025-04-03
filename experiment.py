"""
Train and evaluate model to gather data about it's performance.
"""

import argparse
import json
import logging
import pickle
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from configs.data_model import (
    CMAOptimizerName,
    ExperimentConfig,
    GradientOptimizerConfig,
    DatasetName
)
from models.mlp_classifier import MLPClassifier
from training.train_cmaes import train_cmaes
from training.train_gradient import train_gradient
from training.train_layerwise import train_cmaes_layerwise
from data.iris_dataset import load_iris_dataset
from data.mnist_dataset import load_mnist_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="Path to JSON config file.")
    parser.add_argument('--runs', type=int, default=25, help='Number of model trainings to perform.')
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
            train_dataset, val_dataset, test_dataset = load_iris_dataset(config.train_val_test_ratios)
        case DatasetName.MNIST:
            train_dataset, val_dataset, test_dataset = load_mnist_dataset(config.train_val_test_ratios)
        case _:
            raise ValueError(f"Invalid dataset name {config.dataset_name.name}!")
        
    # dictionary to store statistics
    stats = {
        'time': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'test_loss': [],
        'test_acc': [],
        'model_evals': [],
        'grad_evals': []
    }

    # choose the right training function depending on configuration
    if isinstance(config.optimizer_config, GradientOptimizerConfig):
        train_function = train_gradient
    elif config.optimizer_config.name == CMAOptimizerName.CMAES:
        train_function = train_cmaes
    elif config.optimizer_config.name == CMAOptimizerName.LAYERWISE_CMAES:
        train_function = train_cmaes_layerwise
    else:
        raise ValueError(f"Invalid training method {config.optimizer_config.name}")


    # Create an instance of classfier to train
    model = MLPClassifier(
        input_dim=config.num_features,
        hidden_dim=config.num_hidden,
        output_dim=config.num_classes,
    )

    # Train the model
    for run_num in range(args.runs):
        logger.info(f"Starting run {run_num+1}/{args.runs}")

        start_time = time.time()
        model = train_function(model, train_dataset, val_dataset, config, logger)
        elapsed_time = time.time() - start_time

        model_evals = model.eval_counter
        grad_evals = model.grad_counter

        loss_fn = nn.CrossEntropyLoss()

        train_loss, train_acc = model.evaluate(DataLoader(train_dataset), loss_fn)
        val_loss, val_acc = model.evaluate(DataLoader(val_dataset), loss_fn)
        test_loss, test_acc = model.evaluate(DataLoader(test_dataset), loss_fn)
