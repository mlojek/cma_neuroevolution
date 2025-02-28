"""
Training of model with gradient descent methods.
"""

import torch
from torch import nn, optim

from ..data.iris_loader import load_iris_dataset
from ..models.mlp import MLP
from ..utils.wandb_utils import init_wandb, log_metrics


def train_adam(epochs=50, lr=0.01, batch_size=16, log_interval=5, use_wandb=True):
    # Load data
    train_loader, test_loader = load_iris_dataset(batch_size=batch_size)

    # Initialize model, loss function, and optimizer
    model = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

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

        # Log results
        if use_wandb:
            log_metrics(epoch, avg_loss, accuracy)

        if epoch % log_interval == 0 or epoch == epochs:
            print(
                f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}"
            )

    print("Training complete.")


if __name__ == "__main__":
    train_adam()
