"""
Weights and Biases integration scripts.
https://docs.wandb.ai/quickstart/
"""

import wandb
from data_model import ExperimentConfig


def init_wandb(run_name, config: ExperimentConfig):
    wandb.init(
        project="cma_neuroevolution", name=run_name, config=config
    )


def log_metrics(epoch, loss, accuracy):
    # epoch, [train, test] x [loss, config]
    wandb.log({"epoch": epoch, "loss": loss, "accuracy": accuracy})
