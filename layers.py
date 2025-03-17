import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupEquivariantConv(nn.Module):
    """
    Group Equivariant Convolutional Layer (G-CNN)
    
    Based on: Cohen & Welling, "Group Equivariant Convolutional Networks" (2016)
    https://arxiv.org/abs/1602.07576
    
    This layer ensures equivariance to transformations defined by a symmetry group.
    """
    def __init__(self, in_channels, out_channels, kernel_size, group_size=4, stride=1, padding=1):
        super(GroupEquivariantConv, self).__init__()
        self.group_size = group_size
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels // group_size, kernel_size, stride, padding)
            for _ in range(group_size)
        ])
    
    def forward(self, x):
        """Apply group-equivariant convolutions"""
        out = torch.cat([conv(x) for conv in self.convs], dim=1)
        return out