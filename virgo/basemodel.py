""""""

import numpy as np
from virgo.cluster import VirgoCluster


class BaseModel:
    """Virgo base model for clustering and mixture models."""

    def __init__(
        self,
        vcluster: VirgoCluster,
        fit_dim_ind: list = None,
    ):
        self._vcluster = vcluster
        self._fit_dim_ind = fit_dim_ind
        if self._fit_dim_ind is not None:
            self._data = self._vcluster.scaled_data[:, self._fit_dim_ind]
        else:
            self._data = self._vcluster.scaled_data

        self.model = None
        self.predictions = None

    def fit(self):
        """"""

        self.model.fit(self._data)
        try:
            elbo = self.model.lower_bound_
        except AttributeError:
            elbo = None

        return elbo

    def predict(
        self,
        data: np.array = None,
        remove_uncertain_labels: bool = False,
        uncertainty_prob: float = 0.95,
    ) -> np.ndarray:
        """"""

        if data is None:
            try:
                self.predictions = self.model.predict(self._data)
            except AttributeError:
                self.predictions = self.model.fit_predict(self._data)
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
