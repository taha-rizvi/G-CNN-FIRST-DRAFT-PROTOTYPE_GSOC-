import pytest
import torch
import deepchem as dc
import numpy as np
import unittest
from flaky import flaky
try:
    import torch  # noqa: F401
    from deepchem.models.torch_models.layers import GConv2D
    has_torch = True
except:
    has_torch = False


class TestGCNN(unittest.TestCase):

    @pytest.mark.torch
    def test_2d_gcnn_regression(self):
        """Test that a 2D G-CNN can overfit simple regression datasets."""
        n_samples = 10
        n_features = 3
        n_tasks = 1

        np.random.seed(123)
        X = np.random.rand(n_samples, 10, 10, n_features)
        y = np.random.rand(n_samples, n_tasks).astype(np.float32)
        dataset = dc.data.NumpyDataset(X, y)

        regression_metric = dc.metrics.Metric(dc.metrics.mean_squared_error)
        model = dc.models.torch_models.g_equivariant_nn.GEquivariantNN(
            n_tasks,
            n_features,
            dims=2,
            dropouts=0,
            kernel_size=3,
            mode='regression',
            learning_rate=0.003,
            layer_class=GConv2D  # Use our G-CNN layer
        )

        model.fit(dataset, nb_epoch=200)
        scores = model.evaluate(dataset, [regression_metric])
        assert scores[regression_metric.name] < 0.1

    @pytest.mark.torch
    def test_2d_gcnn_classification(self):
        """Test that a 2D G-CNN can overfit simple classification datasets."""
        n_samples = 10
        n_features = 3
        n_tasks = 1

        np.random.seed(123)
        X = np.random.rand(n_samples, 10, 10, n_features)
        y = np.random.randint(2, size=(n_samples, n_tasks)).astype(np.float32)
        dataset = dc.data.NumpyDataset(X, y)

        classification_metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
        from deepchem.models.torch_models.g_equivariant_nn import GEquivariantNN
        model =GEquivariantNN(
            n_tasks,
            n_features,
            dims=2,
            dropouts=0,
            kernel_size=3,
            mode='classification',
            learning_rate=0.003,
            layer_class=GConv2D
        )

        model.fit(dataset, nb_epoch=100)
        scores = model.evaluate(dataset, [classification_metric])
        assert scores[classification_metric.name] > 0.9
# @pytest.mark.parametrize("batch_size, in_channels, out_channels, height, width, group", [
#     (4, 3, 16, 32, 32, "C4"),
#     (2, 1, 8, 28, 28, "D4"),
# ])
# def test_gconv2d_layer(batch_size, in_channels, out_channels, height, width, group):
#     """Test GConv2D layer for equivariant convolutions."""
    
#     # Initialize G-CNN Layer
#     gconv = GConv2D(in_channels=in_channels, out_channels=out_channels, group=group, kernel_size=3)
    
#     # Create Random Input Tensor
#     x = torch.randn(batch_size, in_channels, height, width)
    
#     # Forward Pass
#     output = gconv(x)
    
#     # Assertions
#     assert output.shape == (batch_size, out_channels, height, width), \
#         f"Expected output shape ({batch_size}, {out_channels}, {height}, {width}), but got {output.shape}"
    
#     assert isinstance(output, torch.Tensor), "Output should be a PyTorch tensor"
    
#     # Test Equivariance (Output should not change shape for transformed input)
#     transformed_x = torch.rot90(x, k=1, dims=(-2, -1))  # Rotate input by 90 degrees
#     transformed_output = gconv(transformed_x)
#     assert transformed_output.shape == output.shape, "Equivariance test failed: output shape changed after transformation"