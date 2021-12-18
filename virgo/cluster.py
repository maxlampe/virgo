"""Central virgo cluster data class"""

import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# np.set_printoptions(edgeitems=10)
# np.core.arrayprint._line_width = 180

# TODO: numpy or pytorch data format?
# ToDo: Use Sklearn? I don't like hybrid packages and GPyTorch is a given
# ToDo: Spatial dimensions fixed input data dim [n_data,  (x, y, z, ...)] (no 2D)


class VirgoCluster:
    """"""

    def __init__(
        self, file_name: str, shuffle_data: bool = True, n_max_data: int = None
    ):
        self._fname = file_name
        self.data = self._load_data(self._fname, shuffle=shuffle_data, n_max=n_max_data)
        self.scaler = None
        self.scaled_data = None

        self.cluster = None
        self.cluster_labels = None

    def scale_data(self):
        """Create second data set and rescale it."""

        scaled_data = self.data[:, 1:]
        self.scaler = StandardScaler()
        self.scaler.fit(scaled_data)
        scaled_data = self.scaler.transform(scaled_data)
        self.scaled_data = scaled_data

    def print_datastats(self):
        """Print a few simple statistics about the data"""

        for ind, data in enumerate([self.data, self.scaled_data]):
            if data is None:
                continue
            print(f"Data set {ind} - Shape: {data.shape}")
            print(f"Mean / Std: {data.mean():0.3f} / {data.std():0.3f}")
            print(f"Min / Max: {data.min():0.3f} / {data.max():0.3f}")

    def plot_cluster(
        self,
        cluster_label: list = None,
        n_step: int = 1,
        remove_uncertain: bool = True,
        maker_size: float = None,
    ):
        """Print all or subset of clusters in 3D plot."""

        assert self.cluster is not None, "No cluster data set."
        assert self.cluster_labels is not None, "No cluster labels set."

        plot_data = self.cluster[::n_step]
        plot_label = self.cluster_labels[::n_step]

        if remove_uncertain:
            uncertain_mask = plot_label >= 0
            plot_data = plot_data[uncertain_mask]
            plot_label = plot_label[uncertain_mask]

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(projection="3d")

        if cluster_label is not None:
            for target_ind, target_label in enumerate(cluster_label):
                curr_data = plot_data[plot_label == target_label]
                curr_label = plot_label[plot_label == target_label]
                if target_ind == 0:
                    plot_data_filt = curr_data
                    plot_label_filt = curr_label
                else:
                    plot_data_filt = np.concatenate([plot_data_filt, curr_data])
                    plot_label_filt = np.concatenate([plot_label_filt, curr_label])

            plot_data = plot_data_filt
            plot_label = plot_label_filt

        if maker_size is None:
            maker_size = 6.0

        ax.scatter(
            plot_data.T[1],
            plot_data.T[2],
            plot_data.T[3],
            c=plot_label,
            marker=".",
            cmap="plasma",
            s=maker_size,
        )
        plt.show()

    def get_labels(self, return_counts=False):
        """Returns available labels and counts per label"""

        assert self.cluster_labels is not None, "No cluster labels set."
        avail_labels = np.unique(self.cluster_labels, return_counts=return_counts)

        return avail_labels

    def sort_labels(self):
        """Relabels cluster dependent on size."""

        assert self.cluster_labels is not None, "No cluster labels set."

        unique = self.get_labels(return_counts=True)
        labels_cp = copy.copy(self.cluster_labels)

        j = 0
        for i in np.argsort(unique[1])[::-1]:
            curr_label = unique[0][i]
            if curr_label >= 0:
                labels_cp[self.cluster_labels == curr_label] = j
                j += 1

        self.cluster_labels = labels_cp

    @staticmethod
    def _load_data(file_name: str, shuffle: bool = True, n_max: int = None):
        """"""

        data = np.loadtxt(file_name)
        n_data = data.shape[0]
        ev_no = np.linspace(0, n_data - 1, n_data, dtype=int)

        # adding event number dimension (ev_no + 2 = line number in file)
        ev_no = np.expand_dims(ev_no, axis=1)
        data = np.append(data, ev_no, axis=1)
        ind_list = list(range(data.shape[1]))
        ind_list.insert(0, ind_list.pop(len(ind_list) - 1))
        data = data[:, ind_list]

        if shuffle:
            np.random.shuffle(data)
        if n_max is not None:
            data = data[:n_max]

        return data


def main():
    file_name = "/home/max/Software/virgo/data/data.txt"
    cluster = VirgoCluster(file_name=file_name)
    cluster.scale_data()
    cluster.print_datastats()


if __name__ == "__main__":
    main()
