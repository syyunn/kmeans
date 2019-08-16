import logging
import argparse

from Tester import Tester
from KMeans import KMeans

import utils


def main():
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument('-n_clusters', type=int, default=5)
    parser.add_argument('-n_points', type=int, default=100)

    opt = parser.parse_args()

    tester = Tester(n_gaussian_clusters=opt.n_clusters)

    # Generate data from n 2d multivariate gaussian parameters
    data, labels = tester.generate_2d_gaussian_points(
        how_many_per_each_gaussian=opt.n_points)
    logging.info(" Generated {} data points from {} different 2 dimensional "
                 "multivariate gaussian distributions. ({} data points for "
                 "each cluster.)".format(opt.n_clusters * opt.n_points,
                                         opt.n_clusters, opt.n_points))

    # Raw Data
    utils.draw(data, labels, without_label_color=True, means=None,
               title="Data", save="result/raw.png", show=False)
    utils.draw(data, labels, without_label_color=False, means=tester.means,
               title="Gaussian", save="result/gaussian.png", show=False)

    # KMeans Prediction
    kmeans = KMeans(n_cluster=opt.n_clusters)
    prediction_lables, prediction_centers = kmeans.fit(data)
    utils.draw(data, prediction_lables, without_label_color=False,
               means=prediction_centers, title="KMeans",
               save="result/kmeans.png", show=False)

    # Concatenate results
    png_list = ["result/raw.png", "result/gaussian.png", "result/kmeans.png"]
    utils.concatenate_pngs(png_list, "result/final.png")


if __name__ == "__main__":
    main()
