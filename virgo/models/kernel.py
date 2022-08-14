"""Virgo kernel class for creating new feature space dimensions."""

from virgo.data.cluster import VirgoCluster
from typing import Callable
import numpy as np
import torch
from gpytorch import kernels

from scipy.linalg import svd
from sklearn.decomposition import PCA


class BaseKernel:
    """"""

    def __init__(self, vcluster: VirgoCluster, spatial_dim: list = [0, 1, 2]):
        self._vcluster = vcluster
        self._spatial_dim = spatial_dim

    def __call__(self, kernel_func: Callable = None, **kwargs):
        x_in = self._vcluster.scaled_data
        assert x_in is not None, "Scaled data is None."

        kernel_space = self.calc_kernel_space(x_in, kernel_func, **kwargs)
        self._vcluster.scaled_data = kernel_space

    def calc_kernel_space(
        self, x_data: np.array, kernel_func: Callable = None
    ) -> np.array:
        """"""
        pass


class VirgoSimpleKernel(BaseKernel):
    """"""

    def __init__(self, vcluster: VirgoCluster, spatial_dim: list = [0, 1, 2]):
        super().__init__(vcluster=vcluster, spatial_dim=spatial_dim)
        self._covar_module = kernels.LinearKernel()
        self._covar_module2 = kernels.RBFKernel()

    def calc_kernel_space(self, x_data: np.array) -> np.array:
        torch_subdata = torch.tensor(x_data[:, self._spatial_dim])
        lazy_covar_matrix = self._covar_module(torch_subdata)
        z_vals = 10.0 * lazy_covar_matrix.diag().detach().numpy()

        return np.array([z_vals, *x_data.T]).T


class VirgoKernel(BaseKernel):
    """"""

    def __init__(
        self,
        vcluster: VirgoCluster,
        spatial_dim: list = [0, 1, 2],
        k_nystroem: int = 500,
        pca_comp: int = 10,
        add_dim_back: int = None,
    ):
        super().__init__(vcluster=vcluster, spatial_dim=spatial_dim)
        self._k_nystroem = k_nystroem
        self._pca_comp = pca_comp
        self._add_dim_back = add_dim_back

    def calc_kernel_space(
        self, x_data: np.array, kernel_func: Callable = None
    ) -> np.array:
        """"""

        if kernel_func is None:
            # default
            kernel = kernels.RBFKernel()
            kernel.lengthscale = 2 * x_data[:, self._spatial_dim].var()
        else:
            kernel = kernel_func

        kernel_space = self.nystroem_transformation_gpytorch(
            x_data[:, self._spatial_dim], k=self._k_nystroem, kernel_function=kernel
        )

        kernel_space_pca = PCA(n_components=self._pca_comp).fit_transform(kernel_space)

        if self._add_dim_back is not None:
            kernel_space_pca = np.array(
                [x_data.T[self._add_dim_back], *kernel_space_pca.T]
            ).T

        return kernel_space_pca

    @staticmethod
    def custom_kernel(a, b, gamma=2.0, alpha=2.0):
        spat_a = a[:, [0, 1, 2]]
        spat_b = b[:, [0, 1, 2]]
        norm_a = a[:, [3, 4, 5]]
        norm_b = b[:, [3, 4, 5]]

        kernel = kernels.MaternKernel()
        kernel.lengthscale = gamma
        kernel2 = kernels.LinearKernel()

        return (
            kernel(spat_a, spat_b) * kernel2(spat_a, spat_b)
            + kernel(spat_a, spat_b) * kernel2(norm_a, norm_b)
            + kernel(spat_a, spat_b) * kernel2(norm_a, norm_b)
        )

    @staticmethod
    def nystroem_transformation_gpytorch(x_data, k, kernel_function):
        """Mapping to approximated kernel space"""

        random_sample_indices = np.random.choice(x_data.shape[0], k)
        torch_data = torch.tensor(x_data)
        C = (
            kernel_function(torch_data, torch_data[random_sample_indices])
            .evaluate()
            .detach()
            .numpy()
        )

        W = C[random_sample_indices]
        # Calculating mapping_matrix = W^(-1/2)
        U, S, V = svd(W)
        mapping_matrix = np.linalg.pinv(U * np.sqrt(S) @ V).T

        return C @ mapping_matrix
