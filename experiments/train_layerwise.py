"""
Training of model with CMA-ES optimization by using a separate CMA-ES optimizer
for each set of learnable parameters.
"""

# pylint: disable=too-many-arguments, too-many-locals

from logging import Logger

import cma
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.mlp_classifier import MLPClassifier
from utils.wandb_utils import init_wandb, log_training_metrics


def train_cmaes_layerwise(
    model: MLPClassifier,
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    epochs: int,
    *,
    sigma: float = 1,
    batch_size: int = 16,
    use_wandb: bool = False,
    logger: Logger = None,
    popsize: int = 10,
) -> MLPClassifier:
    """
    Train the MLP classifier using the CMA-ES optimization method, with a separate CMA-ES
    optimizer for each set of learnable parameters.

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
        popsize (int): Number of solutions per iteration.

    Returns:
        MLPClassifier: Trained classifier model.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    loss_function = nn.CrossEntropyLoss()

    # setup CMA-ES optimizer
    model_layer_params = model.get_params_layers()
    optimizers = [
        cma.CMAEvolutionStrategy(param_vector, sigma, {"popsize": popsize})
        for param_vector in model_layer_params
    ]

    if use_wandb:
        init_wandb("layerwise_cma_es", {})

    with torch.no_grad():
        model.eval()
        for epoch in range(epochs):
            # training step
            train_loss = 0
            train_correct_samples = 0
            train_num_samples = 0

            params_backup = model.get_params_layers()
            for x_batch, y_batch in train_loader:
                for layer_idx, es in enumerate(optimizers):
                    solutions = es.ask()
                    losses = []

                    for new_params in solutions:
                        params = model.get_params_layers()
                        params[layer_idx] = new_params
                        model.set_params_layers(params)
                        losses.append(
                            model.evaluate_batch(x_batch, y_batch, loss_function)[0]
                        )

                    es.tell(solutions, losses)
                    model.set_params_layers(params_backup)

                # Set each layer to the best found solution
                best_params = [es.best.x for es in optimizers]
                model.set_params_layers(best_params)

                # Training loss and accuracy
                y_predicted = model(x_batch)

                train_loss += loss_function(y_predicted, y_batch).item() * x_batch.size(
                    0
                )

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
