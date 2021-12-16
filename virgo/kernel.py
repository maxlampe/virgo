"""Virgo kernel class for creating new feature space dimensions."""

from virgo.cluster import VirgoCluster
import numpy as np
import torch
from gpytorch import kernels

# ToDo: Needs to be expand for different kernels and different feature dims
# ToDo: Currently all over the place: GPyTorch, Sklearn, Torch, ...
# ToDo: Currently only uses rescaled data scaled_data
# ToDo: Good idea to just append dim to scaled_data?


class VirgoKernel:
    """"""

    def __init__(self, cluster: VirgoCluster, spatial_dim: list = [0, 1, 2]):
        self._vcluster = cluster
        self._spatial_dim = spatial_dim
        self._covar_module = kernels.LinearKernel()

    def __call__(self):
        x_in = self._vcluster.scaled_data
        assert x_in is not None, "Scaled data is None."

        torch_subdata = torch.tensor(x_in[:, [0, 1, 2]])
        lazy_covar_matrix = self._covar_module(torch_subdata)
        z_vals = lazy_covar_matrix.diag().detach().numpy()

        self._vcluster.scaled_data = np.array([*x_in.T, z_vals]).T
