"""
Script to prepare dataset pickles. It splits data into train, val and test parts and save each
to a pickle file.
"""

import argparse
import pickle

from data.iris_dataset import load_iris_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python prepare_data.py",
        description="Script to prepare dataset pickle files.",
    )
    args = parser.parse_args()

    dataset_splits = load_iris_dataset((0.6, 0.2, 0.2))

    for split_name, split_data in zip(["train", "val", "test"], dataset_splits):
        with open(f"iris.{split_name}.pkl", "wb") as pickle_handle:
            pickle.dump(split_data, pickle_handle)
