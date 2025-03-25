# ---- src/model.py ----
import torch
import torch.nn as nn

class AcupunctureModel(nn.Module):
    """
    A feedforward neural network for classifying acupuncture points based on TCM features.
    """
    def __init__(self, input_dim, output_dim):
        """
        Initialize the model layers.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output classes (acupoints).
        """
        super(AcupunctureModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)

        Returns:
            torch.Tensor: Output probabilities for each class
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

if __name__ == '__main__':
    # Temporary CLI block for testing the model definition
    input_dim = 101  # Update if your input dimension changes
    output_dim = 5   # Number of acupoint classes
    model = AcupunctureModel(input_dim, output_dim)
    print(model)
