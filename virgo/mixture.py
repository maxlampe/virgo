"""Gaussian mixture model for unsupervised classification."""


from virgo.cluster import VirgoCluster
from virgo.basemodel import BaseModel
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import OPTICS, DBSCAN, SpectralClustering, AgglomerativeClustering


class VirgoMixture(BaseModel):
    """"""

    def __init__(
        self,
        vcluster: VirgoCluster,
        n_comp: int = 10,
        mixture_type: str = "gaussian",
        fit_dim_ind: list = None,
    ):
        super().__init__(vcluster=vcluster, fit_dim_ind=fit_dim_ind)
        self._n_comp = n_comp
        self._mixture_type = mixture_type

        self._valid_mixtures = ["gaussian", "bayesian_gaussian"]
        if self._mixture_type not in self._valid_mixtures:
            raise ValueError(f"Invalid mixture type. Not in {self._valid_mixtures}")

        elif self._mixture_type == self._valid_mixtures[0]:
            self.model = GaussianMixture(n_components=n_comp)
        elif self._mixture_type == self._valid_mixtures[1]:
            self.model = BayesianGaussianMixture(n_components=n_comp)


class VirgoClustering(BaseModel):
    """"""

    def __init__(
        self,
        vcluster: VirgoCluster,
        min_samples: int = 10,
        n_clusters: int = 10,
        clustering_type: str = "optics",
        fit_dim_ind: list = None,
    ):
        super().__init__(vcluster=vcluster, fit_dim_ind=fit_dim_ind)
        self._min_samples = min_samples
        self._n_clusters = n_clusters
        self._clustering_type = clustering_type

        self._valid_clustering = ["optics", "dbscan", "spectral", "agglo"]
        if self._clustering_type not in self._valid_clustering:
            raise ValueError(
                f"Invalid clustering type. Not in {self._valid_clustering}"
            )

        elif self._clustering_type == self._valid_clustering[0]:
            self.model = OPTICS(min_samples=50)
        elif self._clustering_type == self._valid_clustering[1]:
            eps = 0.05
            min_samples = 5
            print(eps, min_samples)
            self.model = DBSCAN(min_samples=min_samples, eps=eps)
        elif self._clustering_type == self._valid_clustering[2]:
            self.model = SpectralClustering(n_clusters=self._n_clusters)
        elif self._clustering_type == self._valid_clustering[3]:
            self.model = AgglomerativeClustering(
                n_clusters=self._n_clusters, linkage="ward"
            )
