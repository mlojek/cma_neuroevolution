"""
Model training script.
"""

import argparse
import logging
import pickle

import torch

from experiments.train_cmaes import train_cmaes
from experiments.train_gradient import train_gradient
from models.mlp_classifier import MLPClassifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "method", choices=["adam", "sgd", "cmaes"], help="Optimization type."
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

    with open("iris.train.pkl", "rb") as pickle_handle:
        train_dataset = pickle.load(pickle_handle)

    with open("iris.val.pkl", "rb") as pickle_handle:
        val_dataset = pickle.load(pickle_handle)

    with open("iris.test.pkl", "rb") as pickle_handle:
        test_dataset = pickle.load(pickle_handle)

    model = MLPClassifier()

    if args.method in ["adam", "sgd"]:
        model = train_gradient(
            model,
            train_dataset,
            val_dataset,
            epochs=50,
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
            epochs=50,
            batch_size=16,
            use_wandb=args.use_wandb,
            logger=logger,
        )
    else:
        raise ValueError(f"{args.method} is not a valid training method!")

    torch.save(model.state_dict(), f"iris.{args.method}.pth")
