"""
Training of model with CMA-ES optimization.
"""

# pylint: disable=too-many-arguments, too-many-locals

from logging import Logger

import cma
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.mlp_classifier import MLPClassifier
from utils.wandb_utils import init_wandb, log_training_metrics


def evaluate(model, val_loader, loss_function, params):
    """Evaluate model performance on the validation set."""
    model.set_params(params)
    val_loss = 0.0
    val_correct_samples = 0
    val_num_samples = 0

    for x_val, y_val in val_loader:
        y_predicted = model(x_val)
        val_loss += loss_function(y_predicted, y_val).item() * x_val.size(0)
        predicted_labels = torch.max(y_predicted, 1)[1]
        val_correct_samples += (predicted_labels == y_val).sum().item()
        val_num_samples += y_val.size(0)

    return val_loss / val_num_samples


def train_cmaes(
    model: MLPClassifier,
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    epochs: int,
    *,
    sigma: float = 1,
    batch_size: int = 16,
    use_wandb: bool = False,
    logger: Logger = None,
) -> MLPClassifier:
    """
    Train the MLP classifier using the CMA-ES optimization method.

    Args:
        model (MLPClassifier): The model to train.
        train_dataset (TensorDataset): Training split of the dataset.
        val_dataset (TensorDataset): Validation split of the dataset.
        epochs (int): Number of training epochs.
        sigma (float): Initial standard deviation for CMA-ES.
        batch_size (int): Number of samples per batch, default 16.
        use_wandb (bool): If true, loss and accuracy metrics will be logged
            to wandb.ai, default False.
        logger (Logger): Logger to log training and validation metrics.

    Returns:
        MLPClassifier: Trained classifier model.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    loss_function = nn.CrossEntropyLoss()

    # setup CMA-ES optimizer
    es = cma.CMAEvolutionStrategy(x0=model.params_to_tensor(), sigma0=sigma)

    if use_wandb:
        init_wandb("whole_model_cma_es", {})

    with torch.no_grad():
        model.eval()
        for epoch in range(epochs):
            # training step
            train_loss = 0
            train_correct_samples = 0
            train_num_samples = 0

            # ==== experimental start ====
            for x_batch, y_batch in train_loader:
                solutions = es.ask()
                losses = [
                    evaluate(model, train_loader, loss_function, params)
                    for params in solutions
                ]
                es.tell(solutions, losses)
                best_params = es.best.x
                model.set_params(best_params)

                y_predicted = model(x_batch)
                train_loss = loss_function(y_predicted, y_batch)

                train_loss += train_loss.item() * x_batch.size(0)

                predicted_labels = torch.max(y_predicted, 1)[1]

                train_correct_samples += (predicted_labels == y_batch).sum().item()
                train_num_samples += y_batch.size(0)  # ==== experiemental end ====

            # Compute train loss and accuracy
            train_avg_loss = train_loss / train_num_samples
            train_accuracy = train_correct_samples / train_num_samples

            # Validation step
            val_loss = 0
            val_correct_samples = 0
            val_num_samples = 0

            for x_val, y_val in val_loader:
                y_predicted = model(x_val)

                val_loss += loss_function(y_predicted, y_val).item() * x_val.size(0)

                predicted_labels = torch.max(y_predicted, 1)[1]
                val_correct_samples += (y_predicted == y_val).sum().item()
                val_num_samples += y_val.size(0)

            # Validation loss and accuracy
            val_avg_loss = val_loss / val_num_samples
            val_accuracy = val_correct_samples / val_num_samples

            if logger:
                logger.info(
                    f"Epoch {epoch+1}/{epochs}: "
                    f"train loss: {train_avg_loss:.4f}, train accuracy: {train_accuracy:.4f}, "
                    f"val loss: {val_avg_loss:.4f}, val accuracy: {val_accuracy:.4f}"
                )

            if use_wandb:
                log_training_metrics(
                    epoch + 1,
                    train_avg_loss,
                    train_accuracy,
                    val_avg_loss,
                    val_accuracy,
                )

    return model
