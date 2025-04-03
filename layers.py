# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class GroupEquivariantConv(nn.Module):
#     """
#     Group Equivariant Convolutional Layer (G-CNN)
    
#     Based on: Cohen & Welling, "Group Equivariant Convolutional Networks" (2016)
#     https://arxiv.org/abs/1602.07576
    
#     This layer ensures equivariance to transformations defined by a symmetry group.
#     """
#     def __init__(self, in_channels, out_channels, kernel_size, group_size=4, stride=1, padding=1):
#         super(GroupEquivariantConv, self).__init__()
#         self.group_size = group_size
#         self.convs = nn.ModuleList([
#             nn.Conv2d(in_channels, out_channels // group_size, kernel_size, stride, padding)
#             for _ in range(group_size)
#         ])
    
#     def forward(self, x):
#         """Apply group-equivariant convolutions"""
#         out = torch.cat([conv(x) for conv in self.convs], dim=1)
#         return out
import torch
import torch.nn as nn
import deepchem as dc
from groupy.gconv.pytorch_gconv import P4ConvZ2, P4ConvP4

class GCNNModel(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(GCNNModel, self).__init__()
        self.conv1 = P4ConvZ2(in_channels=input_channels, out_channels=16, kernel_size=3)
        self.conv2 = P4ConvP4(in_channels=16, out_channels=32, kernel_size=3)
        self.fc = nn.Linear(32 * 7 * 7, output_dim)  # Adjust based on input size

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# Wrap it in DeepChem
model = dc.models.TorchModel(model=GCNNModel(input_channels=1, output_dim=1), loss=dc.models.losses.L2Loss())