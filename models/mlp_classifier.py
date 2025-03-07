"""
Simple multi-layer perceptron classifier model.
Default layer dimensionality matches the Iris dataset.
"""

import torch
from torch import Tensor, nn


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

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform forward propagation of the model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output values.
        """
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
