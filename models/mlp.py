"""
Simple multi-layer perceptron classifier model.
Default layer dimensionality matches the Iris dataset.
"""

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
