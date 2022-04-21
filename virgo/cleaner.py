"""Different label cleaner and other tools."""

import numpy as np
from virgo.cluster import VirgoCluster
from sklearn.mixture import GaussianMixture


class BaseCleaner:
    """"""

    def __init__(self, vcluster: VirgoCluster):
        self._vcluster = vcluster
        self.clusters = None
        self.labels = None

    def clean(self, sort_labels: bool = True):
        """"""

        self.clusters = []
        self.labels = []
        for target_label in self._vcluster.get_labels():
            mask = self._vcluster.cluster_labels == target_label
            tmp_data = self._vcluster.cluster[mask]
            tmp_label = self._vcluster.cluster_labels[mask]
            tmp_data_clean = None
            tmp_label_clean = None
            if target_label >= 0:
                tmp_data_clean, tmp_label_clean = self._clean_cluster(
                    tmp_data, tmp_label
                )

            if tmp_data_clean is not None and tmp_label_clean is not None:
                self.clusters.append(tmp_data_clean)
                self.labels.append(tmp_label_clean)
            else:
                tmp_label[:] = -1
                self.clusters.append(tmp_data)
                self.labels.append(tmp_label)

        self.clusters = np.array(self.clusters)
        self.labels = np.array(self.labels)

        # This is inefficient
        all_clusters, all_labs = None, None
        for ind, clust in enumerate(self.clusters):
            if ind == 0:
                all_clusters = np.array(clust)
                all_labs = np.array(self.labels[ind])
            else:
                all_clusters = np.concatenate([all_clusters, clust])
                all_labs = np.concatenate([all_labs, self.labels[ind]])

        self._vcluster.cluster = all_clusters
        self._vcluster.cluster_labels = all_labs
        if sort_labels:
            self._vcluster.sort_labels()

    def _clean_cluster(self, tmp_data: np.array, tmp_label: np.array) -> tuple:
        """"""
        pass


class GaussianMixtureCleaner(BaseCleaner):
    """Studies each cluster if it should be separated by fitting two component GM."""

    def __init__(self, vcluster: VirgoCluster):
        super().__init__(vcluster=vcluster)
        self.unique_labels = self._vcluster.get_labels()

    def _clean_cluster(self, tmp_data: np.array, tmp_label: np.array) -> tuple:
        """"""

        model = GaussianMixture(n_components=1)
        model.fit(tmp_data)
        m1 = model.lower_bound_

        model = GaussianMixture(n_components=2)
        model.fit(tmp_data)
        m2 = model.lower_bound_

        print(f" {m1 / m2:0.5f}, {m1:0.5f}, {m2:0.5f}")
        # ToDo: Value empirical, will generalize badly! Probably data scale dependent
        if (m1 / m2) < 1.075:
            return tmp_data, tmp_label
        else:
            new_preds = model.predict(tmp_data)
            new_label = int(self.unique_labels.max() + 1)

            valid_data = tmp_data[new_preds == 0]
            valid_label = tmp_label[new_preds == 0]

            valid_data = np.concatenate([valid_data, tmp_data[new_preds == 1]])
            tmp_label[new_preds == 1] = new_label
            valid_label = np.concatenate([valid_label, tmp_label[new_preds == 1]])

            self.unique_labels = np.append(self.unique_labels, new_label)

            return valid_data, valid_label


class LowDensityCleaner(BaseCleaner):
    """Studies each cluster if it should be separated by fitting two component GM."""

    def __init__(
        self,
        vcluster: VirgoCluster,
        density_threshhold: float
    ):
        super().__init__(vcluster=vcluster)
        self.unique_labels = self._vcluster.get_labels()
        self._density_th = density_threshhold
        self.densities = []

    def _clean_cluster(self, tmp_data: np.array, tmp_label: np.array) -> tuple:
        """"""

        # Need to disregard ev number dim
        cluster_density = self.calc_density(tmp_data[:, 1:])
        self.densities.append(cluster_density)

        if cluster_density <= self._density_th:
            return None, None
        else:
            return tmp_data, tmp_label

    @staticmethod
    def calc_density(cluster: np.array):
        """Simple way of approximating density. Only works in cartesian coordinates."""
        volume = np.abs(cluster.T.max(axis=1) - cluster.T.min(axis=1)).prod()
        n_particles = cluster.shape[0]

        return n_particles / volume


class AutoDensityCleaner:
    """"""

    def __init__(
        self,
        vcluster: VirgoCluster,
        density_threshhold: float = None,
        pick_top: int = 1,
    ):
        self._vcluster = vcluster
        self._density_th = density_threshhold
        self._pick_top = pick_top
        self._gamma = 0.999

    def clean(self):
        """"""

        tmp_cleaner = LowDensityCleaner(self._vcluster, 1.0e-99)
        tmp_cleaner.clean()
        densities = np.array(tmp_cleaner.densities)
        limit_arg = min([self._pick_top, densities.shape[0]])
        top_limit = self._gamma * np.sort(densities)[::-1][limit_arg - 1]

        if self._density_th is not None:
            if top_limit < self._density_th:
                top_limit = self._density_th

        print(f"Density cutoff {top_limit}")
        print(f"Densities: {densities}")
        tmp_cleaner = LowDensityCleaner(self._vcluster, top_limit)
        tmp_cleaner.clean()

