import pyfof
import numpy as np

def _run_fof_for_cluster(pos, linking_length):

    labels = np.zeros( len(pos[:,1]), dtype=np.int )

    print("running FoF")
    # find FoF groups
    groups = pyfof.friends_of_friends(pos, linking_length)
    print("Found ", len(groups), " groups")

    # count the number of particles per group
    Num_entries = np.zeros(len(groups), dtype=np.int)
    for i in range(0,len(groups)):
        Num_entries[i] = len(groups[i])

    # sort and reverse order -> we want largest groups first
    Num_entries_sorted = np.argsort(Num_entries)[::-1]

    # loop over all groups
    for i in range(0, len(groups)):
        # store current active group to make it more readable
        group = groups[Num_entries_sorted[i]]
        # loop over particles in group
        for j in range(0, len(group)):
            # assign ID of current group
            labels[group[j]] = i

    return labels
