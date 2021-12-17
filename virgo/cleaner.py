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

    def clean(self):
        """"""

        self.clusters = []
        self.labels = []
        for target_label in self._vcluster.get_labels():
            print(f"Cluster {target_label}")
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

    def __init__(self, vcluster: VirgoCluster, density_threshhold: float):
        super().__init__(vcluster=vcluster)
        self.unique_labels = self._vcluster.get_labels()
        self._density_th = density_threshhold

    def _clean_cluster(self, tmp_data: np.array, tmp_label: np.array) -> tuple:
        """"""

        cluster_density = self.calc_density(tmp_data)
        print(f"Density: {cluster_density}")
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
