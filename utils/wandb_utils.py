"""
Weights and Biases integration scripts.
https://docs.wandb.ai/quickstart/
"""

import wandb


def init_wandb(run_name, optimizer_name):
    wandb.init(
        project="cma-es-vs-adam", name=run_name, config={"optimizer": optimizer_name}
    )


def log_metrics(epoch, loss, accuracy):
    wandb.log({"epoch": epoch, "loss": loss, "accuracy": accuracy})
