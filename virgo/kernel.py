"""Virgo kernel class for creating new feature space dimensions."""

from virgo.cluster import VirgoCluster
import numpy as np
import torch
from gpytorch import kernels

from scipy.linalg import svd
from sklearn.kernel_approximation import pairwise_kernels
from sklearn.decomposition import PCA

# ToDo: Needs to be expand for different kernels and different feature dims
# ToDo: Currently all over the place: GPyTorch, Sklearn, Torch, ...
# ToDo: Currently only uses rescaled data scaled_data
# ToDo: Good idea to just append dim to scaled_data?


class BaseKernel:
    """"""

    def __init__(self, vcluster: VirgoCluster, spatial_dim: list = [0, 1, 2]):
        self._vcluster = vcluster
        self._spatial_dim = spatial_dim

    def __call__(self):
        x_in = self._vcluster.scaled_data
        assert x_in is not None, "Scaled data is None."

        kernel_space = self.calc_kernel_space(x_in)
        self._vcluster.scaled_data = kernel_space

    def calc_kernel_space(self, x_data: np.array) -> np.array:
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

    def calc_kernel_space(self, x_data: np.array) -> np.array:
        """"""

        # rbf_kernel = lambda A, B: pairwise_kernels(A, B, metric="laplacian", gamma=gamma)
        # rbf_kernel = lambda A, B: pairwise_kernels(A, B, metric="polynomial", gamma=gamma)
        # rbf_kernel = lambda A, B: pairwise_kernels(A, B, metric="cosine")
        # rbf_kernel = lambda A, B: pairwise_kernels(A, B)

        gamma = 1 / (2 * x_data[:, self._spatial_dim].var())
        # rbf_kernel = lambda A, B: pairwise_kernels(A, B, metric="rbf", gamma=gamma)
        rbf_kernel = lambda A, B: pairwise_kernels(A, B, metric="cosine")
        # rbf_kernel = lambda A, B: pairwise_kernels(A, B, metric="polynomial", gamma=gamma)
        # rbf_kernel = lambda A, B: pairwise_kernels(A, B)
        kernel_space = self.nystroem_transformation(
            x_data[:, self._spatial_dim], k=self._k_nystroem, kernel_function=rbf_kernel
        )
        kernel_space_pca = PCA(n_components=self._pca_comp).fit_transform(kernel_space)

        # # gamma = 1 / (2 * x_data[:, [3, 4, 5]].var())
        # rbf_kernel = lambda A, B: pairwise_kernels(A, B)
        # kernel_space = self.nystroem_transformation(
        #     x_data[:, [3, 4, 5]], k=self._k_nystroem, kernel_function=rbf_kernel
        # )
        # kernel_space_pca_2 = PCA(n_components=self._pca_comp).fit_transform(
        #     kernel_space
        # )
        #
        # kernel_space_pca = np.array(
        #     [*kernel_space_pca.T, *kernel_space_pca_2.T]
        # ).T

        if self._add_dim_back is not None:
            kernel_space_pca = np.array(
                [x_data.T[self._add_dim_back], *kernel_space_pca.T]
            ).T

        return kernel_space_pca

    @staticmethod
    def nystroem_approximation(x_data, k, kernel_function):
        """Kernel approximation"""

        random_sample_indices = np.random.choice(x_data.shape[0], size=k)
        C = kernel_function(x_data, x_data[random_sample_indices])
        W = C[random_sample_indices]

        return C, np.linalg.pinv(W)

    @staticmethod
    def nystroem_transformation(x_data, k, kernel_function):
        """Mapping to approximated kernel space"""

        random_sample_indices = np.random.choice(x_data.shape[0], k)
        C = kernel_function(x_data, x_data[random_sample_indices])
        W = C[random_sample_indices]
        # Calculating mapping_matrix = W^(-1/2)
        U, S, V = svd(W)
        mapping_matrix = np.linalg.pinv(U * np.sqrt(S) @ V).T

        return C @ mapping_matrix
