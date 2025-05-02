"""
Script to perform detailed analysis of results.
"""

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
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

    return all(set(df.columns) == all_columns for df in dataframes)


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

    column_order = results[0].columns

    # mean values
    mean_values_df = pd.DataFrame(
        columns=["optimizer", *column_order],
        data=[
            [
                optimizer_name,
                *[result_df[column_name].mean() for column_name in column_order],
            ]
            for result_df, optimizer_name in zip(results, result_names)
        ],
    )

    print("## Mean values")
    print(
        tabulate(
            mean_values_df,
            headers=mean_values_df.columns,
            tablefmt="github",
            floatfmt=".4g",
        ),
        "\n",
    )

    for column_name in column_order:
        print(f"## {column_name}")

        pvalues = np.array(
            mann_whitney_u_test_grid([df[column_name] for df in results])
        )

        if "acc" in column_name:
            pvalues = pvalues.T

        pvalues_str = [
            [f"{value:.3f}" if value is not None else "-" for value in row]
            for row in pvalues
        ]

        this_df = pd.DataFrame(
            columns=["optimizer", "mean", "std", *result_names],
            data=[
                [
                    optimizer_name,
                    result_df[column_name].mean(),
                    result_df[column_name].std(),
                    *pvalues_str[index],
                ]
                for index, (result_df, optimizer_name) in enumerate(
                    zip(results, result_names)
                )
            ],
        )

        print(
            tabulate(
                this_df, headers=this_df.columns, tablefmt="github", floatfmt=".4g"
            ),
            "\n",
        )
