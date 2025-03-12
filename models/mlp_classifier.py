"""
Simple multi-layer perceptron classifier model.
Default layer dimensionality matches the Iris dataset.
"""

from typing import List, Tuple

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

    def get_params(self) -> Tensor:
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

    def get_params_layers(self) -> List[Tensor]:
        """
        Get list of all learnable parameters of the model, each layer flattened to 1D.

        Returns:
            List[Tensor]: List of 1D tensors of all learnable parameters of the model by layer.
        """
        return [param.data.view(-1).detach().numpy() for param in self.parameters()]

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

    def set_params_layers(self, param_vectors: List[Tensor]) -> None:
        """
        Set all learnable parameters to values from a given list of param vectors.

        Args:
            param_vectors (Tensor): List of 1D tensors of parameters to set model parameters to.
        """
        for param, vector in zip(self.parameters(), param_vectors):
            param.data = torch.Tensor(vector).view_as(param)

    def evaluate_batch(
        self, batch_x: Tensor, batch_y: Tensor, loss_function: callable
    ) -> Tuple[float, float]:
        """
        Evaluate the model with a batch of data, that is a series of x and y tensors.
        Return both loss and accuracy of the model.

        Args:
            batch_x (Tensor): Tensors of features.
            batch_y (Tensor): Tensors of classes.
            loss_function (callable): Function used to calculate model's loss value.

        Returns:
            Tuple[float, float]: Model's loss value and accuracy on the batch.
        """
        y_predicted = self(batch_x)

        loss_value = loss_function(y_predicted, batch_y).item() / batch_x.size(0)

        predicted_labels = torch.max(y_predicted, 1)[1]
        accuracy = (predicted_labels == batch_y).sum().item() / batch_y.size(0)

        return loss_value, accuracy

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
