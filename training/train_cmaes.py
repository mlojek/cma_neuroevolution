"""
Training of model with CMA-ES optimization.
"""

from logging import Logger

import cma
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from configs.data_model import ExperimentConfig
from models.mlp_classifier import MLPClassifier
from utils.wandb_utils import init_wandb, log_training_metrics


def train_cmaes(
    model: MLPClassifier,
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    config: ExperimentConfig,
    logger: Logger,
) -> MLPClassifier:
    """
    Train the MLP classifier using the CMA-ES optimization method.

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

    # setup CMA-ES optimizer
    es = cma.CMAEvolutionStrategy(
        model.get_params(),
        config.optimizer_config.sigma0,
        {"popsize": config.optimizer_config.population_size},
    )

    if config.use_wandb:
        init_wandb("whole_model_cma_es", config)

    with torch.no_grad():
        model.eval()
        for epoch in range(config.epochs):
            # Training step
            train_loss = 0
            train_correct_samples = 0
            train_num_samples = 0

            for x_batch, y_batch in train_loader:
                solutions = es.ask()

                losses = []
                for new_params in solutions:
                    model.set_params(torch.Tensor(new_params))
                    losses.append(
                        model.evaluate_batch(x_batch, y_batch, loss_function)[0]
                    )

                es.tell(solutions, losses)

                best_params = es.best.x
                model.set_params(torch.Tensor(best_params))

                y_predicted = model(x_batch)
                loss = loss_function(y_predicted, y_batch)
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
                    f"model evals: {model.eval_counter}, grad evals: 0, "
                    f"train loss: {train_avg_loss:.4f}, train accuracy: {train_accuracy:.4f}, "
                    f"val loss: {val_avg_loss:.4f}, val accuracy: {val_accuracy:.4f}"
                )

            if config.use_wandb:
                log_training_metrics(
                    epoch + 1,
                    model.eval_counter,
                    0,
                    train_avg_loss,
                    train_accuracy,
                    val_avg_loss,
                    val_accuracy,
                )

    return model
