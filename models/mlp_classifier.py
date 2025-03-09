"""
Simple multi-layer perceptron classifier model.
Default layer dimensionality matches the Iris dataset.
"""

from typing import Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader


class MLPClassifier(nn.Module):
    """
    Simple multi-layer perceptron classifier model.
    """

    def __init__(self, input_dim=4, hidden_dim=10, output_dim=3):
        """
        Class constructor.

        Args:
            input_dim (int): Number of input neurons.
            hidden_dim (int): Number of hidden layer neurons.
            output_dim (int): Number of outputs.
        """
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax()

        # number of model evaluations
        self.eval_counter = 0

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform forward propagation of the model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output values.
        """
        self.eval_counter += x.shape[0]

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return self.softmax(x)

    def params_to_tensor(self) -> Tensor:
        """
        Concatenate all learnable parameters into one dimensional tensor.

        Returns:
            Tensor: 1D tensor of all learnable parameters of the model.
        """
        return (
            torch.cat([param.data.view(-1) for param in self.parameters()])
            .detach()
            .numpy()
        )

    def set_params(self, param_vector: Tensor) -> None:
        """
        Set all learnable parameters to values from a given param vector.

        Args:
            param_vector (Tensor): 1D tensor of parameters to set model parameters to.
        """
        offset = 0

        for param in self.parameters():
            num_params = param.numel()
            param.data = torch.tensor(
                param_vector[offset : offset + num_params]
            ).view_as(param)
            offset += num_params

    def evaluate(
        self, loader: DataLoader, loss_function: callable
    ) -> Tuple[float, float]:
        """
        Evaluate the model with data from given dataloader and return
        loss value and accuracy. The evaluation is done in eval mode with
        autograd disabled.

        Args:
            loader (DataLoader): Loader with data.
            loss_function (callable): Loss function.

        Returns:
            Tuple[float, float]: Loss value and model accuracy.
        """
        with torch.no_grad():
            self.eval()

            loss_value = 0
            num_correct_samples = 0
            num_all_samples = 0

            for x, y in loader:
                y_predicted = self(x)

                loss_value += loss_function(y_predicted, y).item() * x.size(0)

                predicted_labels = torch.max(y_predicted, 1)[1]
                num_correct_samples += (predicted_labels == y).sum().item()
                num_all_samples += y.size(0)

            val_avg_loss = loss_value / num_all_samples
            val_accuracy = num_correct_samples / num_all_samples

            return val_avg_loss, val_accuracy
