"""
Script to prepare dataset pickles. It splits data into train, val and test parts and save each
to a pickle file.
"""

import argparse
import json
import pickle
from pathlib import Path

from configs.data_model import DatasetConfig
from data.iris_dataset import load_iris_dataset
from data.mnist_dataset import load_mnist_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python prepare_data.py",
        description="Script to prepare dataset pickle files.",
    )
    parser.add_argument(
        "config", type=Path, help="Path to JSON file with dataset config."
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as file_handle:
        config = DatasetConfig(**json.load(file_handle))

    match config.name:
        case "iris":
            dataset_splits = load_iris_dataset(config.train_val_test_ratios)
        case "mnist":
            dataset_splits = load_mnist_dataset(config.train_val_test_ratios)
        case _:
            raise ValueError(f"Invalid dataset name {config.name}!")

    for split_name, split_data in zip(["train", "val", "test"], dataset_splits):
        assert (
            split_data.tensors[0].shape[1] == config.num_features
        ), f"Split {split_name} feature tensors have different shape than expected!"
        # TODO validate ys
        with open(
            config.save_path / f"{config.name}.{split_name}.pkl", "wb"
        ) as pickle_handle:
            pickle.dump(split_data, pickle_handle)
