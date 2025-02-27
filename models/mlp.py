"""
Simple multi-layer perceptron model. Default layer dimensionality matches the Iris dataset.
"""

from torch import Tensor, nn


class MLP(nn.Module):
    """
    Simple multi-layer perceptron model.
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
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform forward propagation of the model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output values.
        """
        x = self.relu(self.fc1(x))
        return self.fc2(x)
