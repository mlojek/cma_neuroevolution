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

from configs.data_model import ExperimentConfig
from models.mlp_classifier import MLPClassifier
from utils.early_stopping import EarlyStopping
from utils.wandb_utils import init_wandb, log_training_metrics


def train_cmaes_layerwise(
    model: MLPClassifier,
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    config: ExperimentConfig,
    logger: Logger = None,
) -> MLPClassifier:
    """
    Train the MLP classifier using the CMA-ES optimization method, with a separate CMA-ES
    optimizer for each set of learnable parameters.

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

    early_stopping = EarlyStopping(
        config.early_stopping_patience, config.early_stopping_delta
    )

    # setup CMA-ES optimizers
    optimizers = [
        cma.CMAEvolutionStrategy(
            param_vector,
            config.optimizer_config.sigma0,
            {"popsize": config.optimizer_config.population_size},
        )
        for param_vector in model.get_params_layers()
    ]

    if config.use_wandb:
        init_wandb("layerwise_cma_es", config)

    with torch.no_grad():
        model.eval()
        for epoch in range(config.epochs):
            # Training step
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

            # Early stopping
            early_stopping(val_avg_loss, model)

            if early_stopping.stop():
                logger.info(
                    f"Early stopping in epoch {epoch+1} due to lack of improvement."
                )
                model.load_state_dict(early_stopping.best_model_state)
                break

    return model
