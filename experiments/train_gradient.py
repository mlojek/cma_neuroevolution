"""
Training of model with gradient descent methods.
"""

import torch
from torch import nn, optim

from data.iris_loader import load_iris_dataset
from models.mlp import MLP
from utils.wandb_utils import init_wandb, log_metrics


def train_gradient(
    optimizer_name: str,
    epochs=50,
    learning_rate=0.01,
    batch_size=16,
    *,
    log_interval=5,
    use_wandb=False,
) -> nn.Module:
    """
    Train MLP model using gradient methods.

    Args:
        optmizer_name (str): Name of gradient optimization method, choices: [adam, sgd].
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate value.
        batch_size (int): Number of samplex per batch.
        log_interval (int): Interval of epochs to report metrics.
        use_wandb (bool): Wheather to log training progress to wandb.ai.

    Returns:
        nn.Module: Trained MLP model.
    """
    # TODO loaders passed as argments, remove batch_size
    # Load data
    train_loader, test_loader = load_iris_dataset(batch_size=batch_size)

    # Initialize model, loss function, and optimizer
    model = MLP()
    criterion = nn.CrossEntropyLoss()

    match optimizer_name:
        case "adam":
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        case "sgd":
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        case _:
            raise ValueError(f"Invalid optmizer {optimizer_name}")

    # Initialize Weights & Biases
    if use_wandb:
        init_wandb(run_name="adam-run", optimizer_name="Adam")

    # Training loop
    for epoch in range(1, epochs + 1):
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
        if use_wandb:
            log_metrics(epoch, avg_loss, accuracy)

        if epoch % log_interval == 0 or epoch == epochs:
            print(
                f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}"
            )

    print("Training complete.")
    return model
