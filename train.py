import argparse
import json
from pathlib import Path

from data_model import ExperimentConfig
from experiments.train_gradient import train_gradient

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="Path to config file in JSON format.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as file_handle:
        config = ExperimentConfig(**json.load(file_handle))

    print(config)

    train_gradient(config)
