"""
Training of model with gradient descent methods.
"""

# pylint: disable=too-many-arguments, too-many-locals

from logging import Logger

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from configs.data_model import GradientOptimizerName
from models.mlp_classifier import MLPClassifier
from utils.wandb_utils import init_wandb, log_training_metrics


def train_gradient(
    model: MLPClassifier,
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    epochs: int,
    learning_rate: float,
    *,
    batch_size: int = 16,
    use_wandb: bool = False,
    optimizer: str = "adam",
    logger: Logger = None,
) -> MLPClassifier:
    """
    Train the MLP classifier using a gradient optimization method.

    Args:
        model (MLPClassifier): The model to train.
        train_dataset (TensorDataset): Training split of the dataset.
        val_dataset (TensorDataset): Validation split of the dataset.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate value for the optimizer.
        batch_size (int): Number of samples per batch, default 16.
        use_wandb (bool): If true, loss and accuracy metrics will be logged
            to wandb.ai, defualt False.
        optimizer (str): Name of gradient optimizer to use. Available options
            are sgd, adam. Default is Adam.
        logger (Logger): Logger to log training and validation metrics.

    Returns:
        MLPClassifier: Trained classifier model.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    loss_function = nn.CrossEntropyLoss()

    if use_wandb:
        init_wandb(optimizer, {})

    match optimizer:
        case GradientOptimizerName.ADAM:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        case GradientOptimizerName.SGD:
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        case _:
            raise ValueError(f"Invalid optmizer {optimizer}!")

    for epoch in range(epochs):
        # Training step
        model.train()

        train_loss = 0
        train_correct_samples = 0
        train_num_samples = 0

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_predicted = model(x_batch)
            loss = loss_function(y_predicted, y_batch)
            loss.backward()
            optimizer.step()

            # pylint: disable=duplicate-code
            train_loss += loss.item() * x_batch.size(0)

            predicted_labels = torch.max(y_predicted, 1)[1]
            train_correct_samples += (predicted_labels == y_batch).sum().item()
            train_num_samples += y_batch.size(0)

        # Compute train loss and accuracy
        train_avg_loss = train_loss / train_num_samples
        train_accuracy = train_correct_samples / train_num_samples

        # Validation step
        val_avg_loss, val_accuracy = model.evaluate(val_loader, loss_function)

        if logger:
            logger.info(
                f"Epoch {epoch+1}/{epochs}: model evaluations: {model.eval_counter} "
                f"train loss: {train_avg_loss:.4f}, train accuracy: {train_accuracy:.4f}, "
                f"val loss: {val_avg_loss:.4f}, val accuracy: {val_accuracy:.4f}"
            )

        if use_wandb:
            log_training_metrics(
                epoch + 1, train_avg_loss, train_accuracy, val_avg_loss, val_accuracy
            )

    return model
