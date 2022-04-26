"""Central virgo cluster data class"""

import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from virgo.io.binary_reader import read_binary_data
from virgo.fof.run_fof import _run_fof_for_cluster

from matplotlib import animation

# TODO: numpy or pytorch data format?
# ToDo: Use Sklearn? I don't like hybrid packages and GPyTorch is a given
# ToDo: Spatial dimensions fixed input data dim [n_data,  (x, y, z, ...)] (no 2D)


class VirgoCluster:
    """"""

    def __init__(
        self,
        file_name: str,
        io_mode: int = 0,
        mach_floor: float = 1.0,
        mach_ceiling: float = 15.0,
        center=np.zeros(3),
        radius: float = 0.0,  #
        shuffle_data: bool = True,
        cut_mach_dim: int = None,
        n_max_data: int = None,
        random_seed: int = 2022,
    ):
        """
            __init__

        Parameters:
            file_name                   # name of input file
            io_mode                     # how to read the data
                                        #   0: .txt file
                                        #   1: custom binary file
                                        #   2: Gadget snapshot
            mach_floor                  # minimum Mach number to consider
            mach_ceiling                # maximum Mach number to consider
            center = [0, 0, 0]          # center of seleted box (only relevant for io_mode = 2)
            radius = 0                  # radius of selected box (only relevant for io_mode = 2)

        """

        self.rdm_seed = random_seed
        np.random.seed(self.rdm_seed)

        self._fname = file_name
        if self._fname is not None:
            self.data = self._load_data(
                self._fname, io_mode, shuffle=shuffle_data, n_max=n_max_data
            )

        self._cut_mach_dim = cut_mach_dim
        self._mach_floor = mach_floor
        self._mach_ceiling = mach_ceiling

        if self._cut_mach_dim is not None:
            mach_mask = self.data[:, self._cut_mach_dim] < self._mach_ceiling
            self.data = self.data[mach_mask]
            mach_mask = self.data[:, self._cut_mach_dim] >= self._mach_floor
            self.data = self.data[mach_mask]

        self.scaler = None
        self.scaled_data = None

        self.cluster = None
        self.cluster_labels = None

    def filter_dim(self, target_dim: int, ceil: float = None, floor: float = None):
        """Simple filter of data target dimension with ceiling and floor value."""

        if ceil is not None:
            sub_data = self.data[:, target_dim]
            self.data = self.data[sub_data <= ceil]
        if floor is not None:
            sub_data = self.data[:, target_dim]
            self.data = self.data[sub_data > floor]

    def scale_data(self, use_dim: list = None):
        """Create second data set and rescale it."""

        if use_dim is None:
            scaled_data = self.data[:, 1:]
        else:
            # To get rid of added event id on import
            dims = [i + 1 for i in use_dim]
            scaled_data = self.data[:, dims]
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

    def plot_raw_hists(
        self,
        bins: int = 100,
        plot_range: list = None,
        axs_label: list = ["x [c kpc / h]", "y [c kpc / h]", "z [c kpc / h]"],
    ):
        """Visualize raw spatial data as histograms"""

        from matplotlib import colors

        fig, axs = plt.subplots(1, 3, figsize=(10, 3))
        # fig.suptitle("Raw data histograms with LogNorm")

        for i in range(3):
            if plot_range is not None:
                p_range = [plot_range[i % 3], plot_range[(i + 1) % 3]]
            else:
                p_range = None

            axs.flat[i].hist2d(
                self.data[:, i % 3 + 1],
                self.data[:, (i + 1) % 3 + 1],
                bins=bins,
                norm=colors.LogNorm(),
                cmap="plasma",
                range=p_range,
            )
            # to "hide" empty bins
            axs.flat[i].set_facecolor("#0c0887")
            axs.flat[i].set(xlabel=axs_label[i % 3], ylabel=axs_label[(i + 1) % 3])

        plt.subplots_adjust(wspace=0.3)
        plt.tight_layout()
        plt.show()

    def plot_cluster(
        self,
        cluster_label: list = None,
        n_step: int = 1,
        remove_uncertain: bool = True,
        maker_size: float = None,
        plot_kernel_space: bool = False,
        store_gif: bool = False,
        gif_title: str = None,
        axs_label: list = None,
        cmap_vmin: float = None,
        cmap_vmax: float = None,
    ):
        """Print all or subset of clusters in 3D plot."""

        assert self.cluster is not None, "No cluster data set."
        assert self.cluster_labels is not None, "No cluster labels set."

        if plot_kernel_space:
            plot_data = self.scaled_data[::n_step]
            if axs_label is None:
                axs_label = ["phi_0 [ ]", "phi_1 [ ]", "phi_2 [ ]"]
        else:
            # ignore event number dim
            plot_data = self.cluster[::n_step, 1:]
            if axs_label is None:
                axs_label = ["x [c kpc / h]", "y [c kpc / h]", "z [c kpc / h]"]
        plot_label = self.cluster_labels[::n_step]

        if remove_uncertain:
            uncertain_mask = plot_label >= 0
            plot_data = plot_data[uncertain_mask]
            plot_label = plot_label[uncertain_mask]

        if maker_size is None:
            maker_size = 6.0

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

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(projection="3d")

        def animate(i):
            # azimuth angle : 0 deg to 360 deg
            ax.view_init(elev=10, azim=i * 1)
            return (fig,)

        def init():
            ax.scatter(
                plot_data.T[0],
                plot_data.T[1],
                plot_data.T[2],
                c=plot_label,
                marker=".",
                cmap="plasma",
                s=maker_size,
                vmin=cmap_vmin,
                vmax=cmap_vmax,
            )
            ax.set(xlabel=axs_label[0], ylabel=axs_label[1], zlabel=axs_label[2])

        if store_gif:
            # Animate
            ani = animation.FuncAnimation(
                fig, animate, init_func=init, frames=360, interval=100, blit=True
            )
            if gif_title is None:
                file_name = "rotate_azimuth_angle_3d_surf"
            else:
                file_name = gif_title
            ani.save(file_name + ".gif", writer="imagemagick", fps=15)
        else:
            init()

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

    def remove_small_groups(self, remove_thresh: int = None):
        """Remove cluster groups by count threshhold."""

        assert self.cluster_labels is not None, "No cluster labels set."

        unique = self.get_labels(return_counts=True)
        labels_cp = copy.copy(self.cluster_labels)
        label_small = unique[0][unique[1] < remove_thresh]

        for i, label in enumerate(label_small):
            if label >= 0:
                labels_cp[self.cluster_labels == label] = -1

        self.cluster_labels = labels_cp

    def export_cluster(
        self,
        file_name: str = None,
        remove_uncertain: bool = True,
        remove_evno: bool = False,
    ):
        """"""

        if remove_uncertain:
            mask = self.cluster_labels >= 0
            out_cluster = self.cluster[mask]
            out_labels = self.cluster_labels[mask]
        else:
            out_cluster = self.cluster
            out_labels = self.cluster_labels

        if remove_evno:
            out_cluster = out_cluster[:, 1:]

        if file_name is None:
            file_name = "VirgoCluster"

        np.savetxt(f"{file_name}_cluster.txt", out_cluster)
        np.savetxt(f"{file_name}_cluster_labels.txt", out_labels)

    def run_fof(
        self,
        linking_length: float = None,  # 35?
        min_group_size: int = 100,
        use_scaled_data: bool = False,
        n_nn: int = None,
    ):
        """Run simple FoF and assign labels"""

        if use_scaled_data:
            if linking_length is None:
                linking_length = self.get_avg_nn_dist(self.scaled_data, n_nn=n_nn)
                print(f"Estimated ll: {linking_length:0.5f}")
            self.cluster_labels = _run_fof_for_cluster(self.scaled_data, linking_length)
        else:
            if linking_length is None:
                linking_length = self.get_avg_nn_dist(self.data[:, 1:4], n_nn=n_nn)
                print(f"Estimated ll: {linking_length:0.5f}")
            self.cluster_labels = _run_fof_for_cluster(
                self.data[:, 1:4], linking_length
            )
        self.cluster = self.data
        self.remove_small_groups(remove_thresh=min_group_size)
        self.sort_labels()

    @staticmethod
    def get_avg_nn_dist(
        array: np.array, n_samples: int = 10000, n_nn: int = None, label: str = None
    ):
        """Helper function to get the average NN distance from an array."""

        dists = []
        samples = min(n_samples, array.shape[0])
        if n_nn is None:
            n_nn = 20
        n_nn = min(n_nn, array.shape[0] - 1)
        for i in range(samples):
            point = array[i]
            dist_to_all = np.sqrt(((array - point) ** 2).sum(1))
            dist_to_all = np.sort(dist_to_all)
            dists.append(dist_to_all[1 : n_nn + 1].mean())

        dists = np.array(dists)
        # Maybe add cuts for noisy data? To stabilize with mean +- sig?
        # dists = dists[dists < 0.5]
        plt.hist(dists, 100)
        plt.show()
        if label is not None:
            plt.hist(dists, 100, [0.0, 0.5])
            plt.savefig(f"nnd{n_nn}_{label}.png", dpi=300)

        vals, counts = np.unique(dists, return_counts=True)
        mode_value = np.argwhere(counts == np.max(counts))
        mode = vals[mode_value].flatten()[0]
        print(dists.mean(), dists.std(), dists.mean() + 1.5 * dists.std(), mode)
        print(array.shape[0])

        #     return dists.mean()
        return dists.mean()  # + 1.5 * dists.std()

    @staticmethod
    def _load_data(
        file_name: str, io_mode: int, shuffle: bool = True, n_max: int = None
    ):
        """"""

        if io_mode == 0:
            data = np.loadtxt(file_name)
        elif io_mode == 1:
            data = read_binary_data(file_name)
        # elif io_mode == 2:
        # data = read_gadget_data(file_name)
        else:
            assert False, "requested io_mode not implemented!"

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
