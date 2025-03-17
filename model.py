import torch
import torch.nn as nn
from layers import GroupEquivariantConv
import torch.nn.functional as F

# Define a simple G-CNN model using DeepChem's TorchModel
class GCNNModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(GCNNModel, self).__init__()
        self.gcnn1 = GroupEquivariantConv(in_channels, 16, kernel_size=3)
        self.gcnn2 = GroupEquivariantConv(16, 32, kernel_size=3)
        self.fc = nn.Linear(32 * 6 * 6, num_classes)  # Adjust based on feature map size
    
    def forward(self, x):
        x = F.relu(self.gcnn1(x))
        x = F.relu(self.gcnn2(x))
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)

