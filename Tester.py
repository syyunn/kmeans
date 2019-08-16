import numpy as np


def _generate_positive_semidefnite_diagonal_matrix():
    random_matrix = np.random.rand(2, 2)
    cov_matrix = np.dot(random_matrix, random_matrix.transpose())
    cov_matrix[1, 1] = cov_matrix[0, 0]
    cov_matrix[0, 1] = 0
    cov_matrix[1, 0] = 0
    return cov_matrix


class Tester:
    def __init__(self, n_gaussian_clusters=3):
        self.n_gaussian_clusters = n_gaussian_clusters
        self.means = [np.random.uniform(-6, 6, size=2) for _ in
                      range(n_gaussian_clusters)]
        self.covs = [_generate_positive_semidefnite_diagonal_matrix() for _ in
                     range(n_gaussian_clusters)]

        self.how_many_per_each_gaussian = None
        self.data = None
        self.labels = None
        self.n_points = None

    def generate_2d_gaussian_points(self, how_many_per_each_gaussian=10):
        self.how_many_per_each_gaussian = how_many_per_each_gaussian
        self.n_points = int(self.how_many_per_each_gaussian *
                            self.n_gaussian_clusters)
        data_in_label_order = [np.random.multivariate_normal(mean, cov,
                               how_many_per_each_gaussian)
                               for mean, cov in zip(self.means, self.covs)]

        data_in_label_order_np = np.array(data_in_label_order)
        labels = [[label] * self.how_many_per_each_gaussian for label in
                  range(self.n_gaussian_clusters)]
        self.labels = np.array(labels).reshape(self.n_points)
        self.data = data_in_label_order_np.reshape(self.n_points, 2)
        return self.data, self.labels


if __name__ == "__main__":
    tester = Tester(n_gaussian_clusters=2)
    data, labels = tester.generate_2d_gaussian_points(
        how_many_per_each_gaussian=1000)
    pass
