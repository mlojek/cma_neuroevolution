"""
Training of model with gradient descent methods.
"""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from models.mlp_classifier import MLPClassifier
from utils.wandb_utils import init_wandb, log_metrics


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

    Returns:
        MLPClassifier: Trained classifier model.
    """
    match optimizer:
        case "adam":
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        case "sgd":
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        case _:
            raise ValueError(f"Invalid optmizer {optimizer}!")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    loss_function = nn.CrossEntropyLoss()

    # if use_wandb:
    #     init_wandb(run_name="adam-run", optimizer_name="Adam")

    for epoch in range(epochs):
        # Training step
        model.train()

        train_loss = 0
        train_correct_samples = 0
        train_num_samples = 0

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_predicted = model(x_batch)
            train_loss = loss_function(y_predicted, y_batch)
            train_loss.backward()
            optimizer.step()

            train_loss += train_loss.item() * x_batch.size(0)

            predicted_labels = torch.max(y_predicted, 1)[1]
            train_correct_samples += (predicted_labels == y_batch).sum().item()
            train_num_samples += y_batch.size(0)

        # Compute train loss and accuracy
        train_avg_loss = train_loss / train_num_samples
        train_accuracy = train_correct_samples / train_num_samples

        # Validation step
        model.eval()

        val_loss = 0
        val_correct_samples = 0
        val_num_samples = 0

        with torch.no_grad():
            for x_val, y_val in val_loader:
                y_predicted = model(x_val)

                val_loss += loss_function(y_predicted, y_val).item() * x_val.size(0)

                predicted_labels = torch.max(y_predicted, 1)[1]
                val_correct_samples += (y_predicted == y_val).sum().item()
                val_num_samples += y_val.size(0)

        # Validation loss and accuracy
        val_avg_loss = val_loss / val_num_samples
        val_accuracy = val_correct_samples / val_num_samples

        # TODO Log using logger
        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_avg_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
            f"Val Loss: {val_avg_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
        )

        # TODO handle wandb logging
        # if use_wandb:
        #     wandb.log({
        #         'epoch': epoch + 1,
        #         'train_loss': train_avg_loss,
        #         'train_accuracy': train_accuracy,
        #         'val_loss': val_avg_loss,
        #         'val_accuracy': val_accuracy
        #     })

    return model
