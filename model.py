import torch
import torch.nn as nn
import deepchem.models.torch_models.layers as layers
from deepchem.models.torch_models.torch_model import TorchModel
import torch.nn.functional as F 

class GEquivariantNN(nn.Module):
    """
    DeepChem-compatible G-CNN model for molecular property prediction.

    Parameters
    ----------
    input_shape : tuple
        Shape of input tensors.
    num_classes : int
        Number of output classes.
    group : str
        Symmetry group for equivariant convolutions.
    """

    def __init__(self, input_shape, num_classes, group="C4"):
        super(GEquivariantNN, self).__init__()

        self.gconv1 = layers.GConv2D(input_shape[0], 32, kernel_size=3, group=group)
        self.gconv2 = layers.GConv2D(32, 64, kernel_size=3, group=group)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.gconv1(x))
        x = F.relu(self.gconv2(x))
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)



class DeepChemGEquivariantNN(TorchModel):
    """
    DeepChem wrapper for Group Equivariant Convolutional Neural Networks (G-CNNs).
    
    Parameters
    ----------
    input_shape : tuple
        Shape of input tensors.
    num_classes : int
        Number of output classes.
    group : str
        Symmetry group (e.g., "C4", "D4").
    learning_rate : float
        Learning rate for optimization.
    """

    def __init__(self, input_shape, num_classes, group="C4", learning_rate=0.001):
        model = GEquivariantNN(input_shape, num_classes, group)
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        super(DeepChemGEquivariantNN, self).__init__(model, loss=loss, optimizer=optimizer)