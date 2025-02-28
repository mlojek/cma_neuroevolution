"""
Training of model with gradient descent methods.
"""

import torch
from torch import nn, optim

from data.iris_loader import load_iris_dataset
from models.mlp import MLP
from utils.wandb_utils import init_wandb, log_metrics
from data_model import ExperimentConfig, OptimizationType, GradientMethod


def train_gradient(
    config: ExperimentConfig
) -> nn.Module:
    assert config.optimization_type == OptimizationType.GRADIENT
    # TODO loaders passed as argments, remove batch_size
    # Load data
    train_loader, test_loader = load_iris_dataset(config.dataset)

    # Initialize model, loss function, and optimizer
    model = MLP()
    criterion = nn.CrossEntropyLoss()

    match config.optimization_method:
        case GradientMethod.ADAM:
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        case GradientMethod.SGD:
            optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)
        case _:
            raise ValueError(f"Invalid optmizer {config.optimization_method}")

    # Initialize Weights & Biases
    if config.logging.use_wandb:
        init_wandb(run_name="adam-run", optimizer_name="Adam")

    # Training loop
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        total_samples = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(output, 1)
            correct += (predicted == y_batch).sum().item()
            total_samples += y_batch.size(0)

        # Compute epoch loss and accuracy
        avg_loss = total_loss / total_samples
        accuracy = correct / total_samples

        # TODO evaluation on test set

        # Log results
        if config.logging.use_wandb:
            log_metrics(epoch, avg_loss, accuracy)

        if epoch % config.logging.log_interval == 0 or epoch == config.epochs:
            print(
                f"Epoch {epoch}/{config.epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}"
            )

    print("Training complete.")
    return model
