"""
Weights and Biases integration scripts.
https://docs.wandb.ai/quickstart/
"""

import wandb


def init_wandb(run_name: str, config: dict) -> None:
    """
    Initialize wandb integration.

    Args:
        run_name (str): Name of the run - used to differentiate experiments.
        config (dict): Experiment configuration.
    """
    wandb.init(project="cma_neuroevolution", name=run_name, config=config)


def log_training_metrics(
    epoch: int,
    train_loss: float,
    train_accuracy: float,
    val_loss: float,
    val_accuracy: float,
) -> None:
    """
    Log metrics from training of model to wandb.ai.

    Args:
        epoch (int): Number of epoch.
        train_loss (float): Total or average loss value on training data.
        train_accuracy (float): Accuracy of the model on training data.
        val_los (float): Total or average loss value on validation data.
        val_accuracy (float): Accuracy of the model on validation data.
    """
    wandb.log(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
    )
