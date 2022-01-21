import pyfof
import numpy as np


def _run_fof_for_cluster(pos, linking_length):
    """"""

    # find FoF groups
    groups = np.array(pyfof.friends_of_friends(pos, linking_length))
    print("Found ", groups.shape[0], " groups")

    # loop over all groups
    labels = -np.ones(pos.shape[0], dtype=np.int)
    for i, sub_group in enumerate(groups):
        labels[sub_group] = i

    return labels
