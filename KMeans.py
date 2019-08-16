import logging

import numpy as np

from Tester import Tester
import utils


def _check_sanity(data, n_cluster):
    n_points, n_features = data.shape

    if n_points < n_cluster:
        raise ValueError("number of data points must be larger than the "
                         "number of cluster.")
    if n_features != 2:
        raise ValueError("The algorithm is presumably designed for 2 "
                         "dimensional datapoints only.")


def _get_pairwise_distances(X, Y):
    distances = np.sqrt(np.einsum('ijk, ijk-> ji', X-Y, X-Y))
    return distances


def _get_random_seed():
    return np.random.mtrand._rand


def _random_centroid_init(data, n_cluster, random_seed):
    n_points, n_features = data.shape

    if n_points > n_cluster * 3:
        num_init_random_sample = n_cluster * 3 # referenced from sklearn
    else:
        num_init_random_sample = n_points

    init_indices = np.random.randint(0,
                                     n_points,
                                     num_init_random_sample)

    sampled_data = data[init_indices]
    n_sampled_data = sampled_data.shape[0]
    random_centroid_idxs = random_seed.permutation(n_sampled_data)[:n_cluster]
    centers = sampled_data[random_centroid_idxs]

    return centers


def _assign_points_to_cluster(data, centers):
    n_points, _ = data.shape
    n_cluster = centers.shape[0]

    # copy centers/data tensors to compute euclidean distances in parallel
    n_centers_repeated_data = np.stack((data,) * n_cluster, axis=0)
    n_data_repeated_centers = np.stack((centers,) * n_points, axis=1)

    dists_points_to_centers = _get_pairwise_distances(n_centers_repeated_data,
                                                      n_data_repeated_centers)
    dists_to_centers = np.min(dists_points_to_centers, axis=1)
    labels = np.argmin(dists_points_to_centers, axis=1)
    inertia = dists_to_centers.sum()

    return labels, inertia


def _rest_clusters(data, centers, labels, n_cluster):
    new_centers = np.empty(centers.shape, dtype=centers.dtype)
    cluster_members_idxs = [np.where(labels == cluster_idx) for cluster_idx in range(n_cluster)]
    for cluster_idx, cluster_members_idx in enumerate(cluster_members_idxs):
        cluster_specific_data = data[cluster_members_idx]
        new_center = cluster_specific_data.mean(axis=0)
        new_centers[cluster_idx] = new_center
    return new_centers


def _single_kmeans(data, n_cluster, random_seed, max_iter, tolerance):
    n_points, _ = data.shape
    centers = _random_centroid_init(data, n_cluster, random_seed=random_seed)

    for iteration in range(max_iter):
        labels, inertia = _assign_points_to_cluster(data, centers)
        logging.info("[Iter {}] intertia        {:.2f}".format(iteration, inertia))
        new_centers = _rest_clusters(data, centers, labels, n_cluster)
        center_shift = np.linalg.norm(centers - new_centers)
        logging.info("[Iter {}] center_shift    {:.2f}".format(iteration, center_shift))
        if center_shift < tolerance:
            logging.info("[Iter {}] tolerance get reached".format(iteration))
            return labels, centers
        centers = new_centers

    return labels, centers


class KMeans:
    def __init__(self, n_cluster, max_iter=300):
        self.n_cluster = n_cluster
        self.random_seed = _get_random_seed()
        self.max_iter = max_iter
        self.tolerance = 1e-7

    def fit(self, data):
        _check_sanity(data, self.n_cluster)
        labels, centers = _single_kmeans(data, self.n_cluster,
                                         random_seed=self.random_seed,
                                         max_iter=self.max_iter,
                                         tolerance=self.tolerance)
        return labels, centers


if __name__ == "__main__":
    n_cluster = 7
    tester = Tester(n_gaussian_clusters=n_cluster)
    data, labels = tester.generate_2d_gaussian_points(
        how_many_per_each_gaussian=100)
    print(labels)
    print(tester.means)
    utils.draw(data, labels, means=tester.means)

    kmeans = KMeans(n_cluster=n_cluster)
    prediction_lables, prediction_centers = kmeans.fit(data)
    print(prediction_lables)
    print(prediction_centers)
    utils.draw(data, prediction_lables, means=prediction_centers)



