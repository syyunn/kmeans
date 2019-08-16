from itertools import cycle

import numpy as np
import matplotlib.pyplot as plt

from kmeans.Test import Tester


def draw(data, labels, means=None):
    n_clusters = len(np.unique(labels))

    colors = cycle("bgrcmyk")

    fig, ax = plt.subplots()
    cluster_members_idxs = [np.where(labels == cluster_idx)
                            for cluster_idx in range(n_clusters)]

    for cluster_idx in range(n_clusters):
        cluster_member_idxs = cluster_members_idxs[cluster_idx]
        points_in_a_cluster = data[cluster_member_idxs]
        x, y = points_in_a_cluster[:, 0], points_in_a_cluster[:, 1]
        ax.scatter(x, y,
                   c=next(colors),
                   label="cluster_{}".format(cluster_idx),
                   alpha=0.3,
                   edgecolors='none')

        mean_x, mean_y = means[cluster_idx]

        # ax.scatter(mean_x, mean_y, c='black', label="cluster_{}_mean".format(
        #     cluster_idx))
        ax.scatter(mean_x, mean_y, c='black', label=None)


    ax.legend()
    ax.grid(True)

    plt.show()
    plt.close()


if __name__ == "__main__":
    tester = Tester(n_gaussian_clusters=2)
    data, labels = tester.generate_2d_gaussian_points(
        how_many_per_each_gaussian=100)
    draw(data, labels, tester.means)
    pass
