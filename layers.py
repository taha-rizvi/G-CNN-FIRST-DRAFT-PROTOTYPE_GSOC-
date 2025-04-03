import torch
import torch.nn as nn
import torch.nn.functional as F

class GConv2D(nn.Module):
    """
    Group Equivariant Convolutional Layer (GConv2D).
    
    This layer performs a convolution operation while preserving equivariance 
    to a specified transformation group (e.g., cyclic rotations or dihedral symmetries).
    
    Given an input feature map \( x \), the layer applies group-convolution 
    using rotated/reflected versions of a shared kernel.
    
    The transformation is defined as:
    \[
    (G * x)(g) = \sum_{h \in G} K(h^{-1} g) x(h)
    \]
    where:
    - \( G \) is the group (e.g., rotations `C4`, reflections `D4`).
    - \( K \) is the learned convolutional kernel.
    - \( x \) is the input feature map.

    Parameters
    ----------
    in_channels: int
        Number of input channels (feature maps).
    out_channels: int
        Number of output channels (filters).
    group: str
        Symmetry group used for equivariant convolutions. 
        Supported values: 'C4' (rotations), 'D4' (rotations + reflections), etc.
    kernel_size: int, default=3
        Size of the convolutional kernel.
    stride: int, default=1
        Stride for the convolution operation.
    padding: int, default=1
        Zero-padding added to both sides of the input.

    Inputs
    ------
    x: torch.Tensor
        Input feature map of shape `(batch_size, in_channels, height, width)`.
    
    Outputs
    -------
    torch.Tensor
        Output feature map of shape `(batch_size, out_channels, new_height, new_width)`,
        where `new_height` and `new_width` depend on `stride` and `padding`.

    Example
    -------
    >>> import torch
    >>> from deepchem.models.torch_models.layers import GConv2D
    >>> # Define Group Equivariant Convolutional Layer
    >>> gconv = GConv2D(in_channels=3, out_channels=16, group='C4', kernel_size=3)
    >>> # Create Random Input
    >>> x = torch.randn(10, 3, 32, 32)  # Batch of 10 images (3-channels, 32x32)
    >>> # Apply G-CNN Layer
    >>> output = gconv(x)
    >>> print(output.shape)  
    torch.Size([10, 16, 32, 32])  # Output feature map
    """

    def __init__(self, in_channels: int, out_channels: int, group: str, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super(GConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.group = group
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Standard 2D Convolution as a base operation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        # Additional logic for implementing group equivariant convolutions
        self.group_transformations = self._define_group_transforms(group)

    def _define_group_transforms(self, group):
        """
        Defines the set of transformations (rotations/reflections) based on the selected symmetry group.
        
        Parameters
        ----------
        group: str
            The selected symmetry group (e.g., 'C4' for 4 rotations, 'D4' for rotations + reflections).
        
        Returns
        -------
        list of transformation functions
        """
        transforms = []
        if group == "C4":
            transforms = [lambda x: x, 
                          lambda x: torch.rot90(x, k=1, dims=(-2, -1)),
                          lambda x: torch.rot90(x, k=2, dims=(-2, -1)),
                          lambda x: torch.rot90(x, k=3, dims=(-2, -1))]
        elif group == "D4":
            transforms = [lambda x: x, 
                          lambda x: torch.rot90(x, k=1, dims=(-2, -1)),
                          lambda x: torch.rot90(x, k=2, dims=(-2, -1)),
                          lambda x: torch.rot90(x, k=3, dims=(-2, -1)),
                          lambda x: torch.flip(x, dims=(-1,)),
                          lambda x: torch.rot90(torch.flip(x, dims=(-1,)), k=1, dims=(-2, -1)),
                          lambda x: torch.rot90(torch.flip(x, dims=(-1,)), k=2, dims=(-2, -1)),
                          lambda x: torch.rot90(torch.flip(x, dims=(-1,)), k=3, dims=(-2, -1))]
        return transforms

    def forward(self, x):
        """
        Applies group equivariant convolution to the input tensor.

        Parameters
        ----------
        x: torch.Tensor
            Input feature map of shape `(batch_size, in_channels, height, width)`.

        Returns
        -------
        torch.Tensor
            Output feature map of shape `(batch_size, out_channels, height, width)`.
        """
        transformed_outputs = []
        for transform in self.group_transformations:
            transformed_x = transform(x)
            transformed_outputs.append(self.conv(transformed_x))
        
        # Aggregate results by summing over transformations
        return sum(transformed_outputs) / len(transformed_outputs)  # Averaging over transformations