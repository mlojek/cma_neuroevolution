"""
Script to perform detailed analysis of results.
"""

import argparse
import os
from pathlib import Path
from typing import List

import pandas as pd
from optilab.utils.stat_test import mann_whitney_u_test_grid
from tabulate import tabulate

from .config.data_model import CMAOptimizerName, GradientOptimizerName

OPTIMIZERS_ORDER = [
    GradientOptimizerName.SGD.value,
    GradientOptimizerName.ADAM.value,
    CMAOptimizerName.CMAES.value,
    CMAOptimizerName.LAYERWISE_CMAES.value,
]


def validate_same_columns(dataframes: List[pd.DataFrame]) -> bool:
    """
    Given a list of pandas DataFrames, check if they all share the same column set.

    Args:
        dataframes (List[pd.DataFrame]): List of dataframes to check.

    Returns:
        bool: True if all dataframes share the same column set, False otherwise.
    """
    all_columns = set().union(*[set(df.columns) for df in dataframes])

    return all([set(df.columns) == all_columns for df in dataframes])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory to read CSV result files from.",
    )
    args = parser.parse_args()

    result_paths = []

    for file in os.listdir(args.input_dir):
        if file.endswith(".csv"):
            result_paths.append(file)

    # sort to desired order
    result_paths = sorted(
        result_paths,
        key=lambda x: OPTIMIZERS_ORDER.index(x.split(".")[1]),
    )

    results = [
        pd.read_csv(args.input_dir / file_path, index_col=0)
        for file_path in result_paths
    ]

    result_names = [file_path.split(".")[1] for file_path in result_paths]

    # validate if all csvs have the same columns
    assert validate_same_columns(
        results
    ), "Not all result files share the same column names!"

    # TODO create table with mean values
    # TODO for each column
    # TODO perform stat tests for this column with optilab
    # TODO remember to use the right direction of the test (transpose?)
    # TODO append columns with mean and std
    # TODO end for
