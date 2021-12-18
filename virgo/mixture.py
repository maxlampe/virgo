"""Gaussian mixture model for unsupervised classification."""
import numpy as np

from virgo.cluster import VirgoCluster
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture


class VirgoMixture:
    """"""

    def __init__(
        self,
        vcluster: VirgoCluster,
        n_comp: int = 10,
        mixture_type: str = "gaussian",
        fit_dim_ind: list = None,
    ):
        self._vcluster = vcluster
        self._n_comp = n_comp
        self._mixture_type = mixture_type
        self._fit_dim_ind = fit_dim_ind
        if self._fit_dim_ind is not None:
            self._data = self._vcluster.scaled_data[:, self._fit_dim_ind]
        else:
            self._data = self._vcluster.scaled_data

        self.predictions = None

        self._valid_mixtures = ["gaussian", "bayesian_gaussian"]
        if self._mixture_type not in self._valid_mixtures:
            raise ValueError(f"Invalid mixture type. Not in {self._valid_mixtures}")

        elif self._mixture_type == self._valid_mixtures[0]:
            self.model = GaussianMixture(n_components=n_comp)
        elif self._mixture_type == self._valid_mixtures[1]:
            self.model = BayesianGaussianMixture(n_components=n_comp)

    def fit(self):
        """"""

        self.model.fit(self._data)

        return self.model.lower_bound_

    def predict(
        self,
        data: np.array = None,
        remove_uncertain_labels: bool = False,
        uncertainty_prob: float = 0.95,
    ) -> np.ndarray:
        """"""

        if data is None:
            self.predictions = self.model.predict(self._data)
            # set results in data class

            if remove_uncertain_labels:
                pred_probs = self.model.predict_proba(self._data).max(axis=1)
                mask = pred_probs < uncertainty_prob
                self.predictions[mask] = -1
                print(f"Removed {mask.sum()}")

            self._vcluster.cluster = self._vcluster.data
            self._vcluster.cluster_labels = self.predictions
            self._vcluster.sort_labels()

            return self.predictions
        else:
            return self.model.predict(data)
