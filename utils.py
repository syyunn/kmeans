import sys
from itertools import cycle

import PIL
import numpy as np
import matplotlib.pyplot as plt

from Tester import Tester


def draw(data, labels, means=None, without_label_color=False, title=None,
         save=None, show=True):
    n_clusters = len(np.unique(labels))

    colors = cycle("bgrcmyk")

    _, ax = plt.subplots()
    cluster_members_idxs = [np.where(labels == cluster_idx)
                            for cluster_idx in range(n_clusters)]

    for cluster_idx in range(n_clusters):
        if without_label_color:
            color = 'black'
        else:
            color = next(colors)
        cluster_member_idxs = cluster_members_idxs[cluster_idx]
        points_in_a_cluster = data[cluster_member_idxs]
        x, y = points_in_a_cluster[:, 0], points_in_a_cluster[:, 1]
        if means is not None:
            ax.scatter(x, y,
                       c=color,
                       label="cluster_{}".format(cluster_idx+1),
                       alpha=0.3,
                       edgecolors='none')

            mean_x, mean_y = means[cluster_idx]
            ax.scatter(mean_x, mean_y, c='black', label=None)
        else:
            ax.scatter(x, y,
                       c=color,
                       label=None,
                       alpha=0.3,
                       edgecolors='none')

    ax.legend()
    ax.grid(True)
    if title:
        plt.title(title)
    if save:
        plt.savefig(save)
    if show:
        plt.show()
    plt.close()


def concatenate_pngs(png_list, save):
    imgs = [PIL.Image.open(i) for i in png_list]
    min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
    imgs_comb = np.hstack((np.asarray(i.resize(min_shape)) for i in imgs))
    imgs_comb = PIL.Image.fromarray(imgs_comb)
    rgb_im = imgs_comb.convert('RGB')
    rgb_im.save(save)


if __name__ == "__main__":
    tester = Tester(n_gaussian_clusters=2)
    data, labels = tester.generate_2d_gaussian_points(
        how_many_per_each_gaussian=100)
    draw(data, labels, tester.means)
    pass
