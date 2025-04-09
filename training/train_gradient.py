"""
Training of model with gradient descent methods.
"""

from logging import Logger

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from configs.data_model import ExperimentConfig, GradientOptimizerName
from models.mlp_classifier import MLPClassifier
from utils.early_stopping import EarlyStopping
from utils.wandb_utils import init_wandb, log_training_metrics


def train_gradient(
    model: MLPClassifier,
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    config: ExperimentConfig,
    logger: Logger,
) -> MLPClassifier:
    """
    Train the MLP classifier using a gradient optimization method.

    Args:
        model (MLPClassifier): The model to train.
        train_dataset (TensorDataset): Training split of the dataset.
        val_dataset (TensorDataset): Validation split of the dataset.
        config (TrainingConfig): Configuration of training hyperparameters.
        logger (Logger): Logger to log training and validation metrics.

    Returns:
        MLPClassifier: Trained classifier model.
    """
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    loss_function = nn.CrossEntropyLoss()

    early_stopping = EarlyStopping(**config.early_stopping.model_dump())

    match config.optimizer_config.name:
        case GradientOptimizerName.ADAM:
            optimizer = optim.Adam(
                model.parameters(), lr=config.optimizer_config.learning_rate
            )
        case GradientOptimizerName.SGD:
            optimizer = optim.SGD(
                model.parameters(), lr=config.optimizer_config.learning_rate
            )
        case _:
            raise ValueError(f"Invalid optmizer {optimizer}!")

    if config.use_wandb:
        init_wandb(f"gradient_{config.optimizer_config.name.value}", config)

    for epoch in range(config.epochs):
        # Training step
        model.train()

        train_loss = 0
        train_correct_samples = 0
        train_num_samples = 0

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_predicted = model(x_batch)
            loss = loss_function(y_predicted, y_batch)
            model.grad_counter += 1
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_batch.size(0)

            predicted_labels = torch.max(y_predicted, 1)[1]
            train_correct_samples += (predicted_labels == y_batch).sum().item()
            train_num_samples += y_batch.size(0)

        # Compute train loss and accuracy
        train_avg_loss = train_loss / train_num_samples
        train_accuracy = train_correct_samples / train_num_samples

        # Validation step
        val_avg_loss, val_accuracy = model.evaluate(val_loader, loss_function)

        if (epoch + 1) % config.log_interval == 0:
            logger.info(
                f"Epoch {epoch+1}/{config.epochs}: "
                f"model evals: {model.eval_counter}, grad evals: {model.grad_counter}, "
                f"train loss: {train_avg_loss:.4f}, train accuracy: {train_accuracy:.4f}, "
                f"val loss: {val_avg_loss:.4f}, val accuracy: {val_accuracy:.4f}"
            )

        if config.use_wandb:
            log_training_metrics(
                epoch + 1,
                model.eval_counter,
                model.grad_counter,
                train_avg_loss,
                train_accuracy,
                val_avg_loss,
                val_accuracy,
            )

        # Early stopping
        early_stopping(val_avg_loss, model)

        if early_stopping.stop():
            logger.info(
                f"Early stopping in epoch {epoch+1} due to lack of improvement."
            )
            model.load_state_dict(early_stopping.best_model_state)
            break

    return model
