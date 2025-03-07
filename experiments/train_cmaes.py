"""
Training of model with CMA-ES optimization.
"""

import cma
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.mlp_classifier import MLPClassifier
from utils.wandb_utils import init_wandb, log_metrics


def evaluate(model, val_loader, loss_function, params):
    """Evaluate model performance on the validation set."""
    model.set_params(params)
    model.eval()
    val_loss = 0.0
    val_correct_samples = 0
    val_num_samples = 0

    with torch.no_grad():
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

    Returns:
        MLPClassifier: Trained classifier model.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    loss_function = nn.CrossEntropyLoss()

    param_vector = model.params_to_tensor()
    es = cma.CMAEvolutionStrategy(param_vector, sigma)

    for epoch in range(epochs):
        solutions = es.ask()
        losses = [
            evaluate(model, train_loader, loss_function, params) for params in solutions
        ]
        es.tell(solutions, losses)
        best_params = es.best.x
        model.set_params(best_params)

        best_loss = min(losses)
        print(f"Epoch {epoch+1}/{epochs} - Val Loss: {best_loss:.4f}")

        # if use_wandb:
        #     wandb.log({"epoch": epoch + 1, "val_loss": best_loss})

    return model
