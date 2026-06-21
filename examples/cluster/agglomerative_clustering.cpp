// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

#include <Skigen/Cluster>

#include <Eigen/Core>

#include <iomanip>
#include <iostream>
#include <random>

namespace {

Eigen::MatrixXd make_hierarchical_clusters() {
    constexpr int samples_per_cluster = 10;
    Eigen::MatrixXd samples(samples_per_cluster * 3, 2);

    std::mt19937 rng(13);
    std::normal_distribution<double> noise(0.0, 0.18);
    const Eigen::Matrix<double, 3, 2> centers = (Eigen::Matrix<double, 3, 2>() <<
        -3.0, 0.0,
         3.0, 0.0,
         0.0, 4.0).finished();

    for (int cluster_index = 0; cluster_index < 3; ++cluster_index) {
        for (int sample_index = 0; sample_index < samples_per_cluster; ++sample_index) {
            const int row = cluster_index * samples_per_cluster + sample_index;
            samples(row, 0) = centers(cluster_index, 0) + noise(rng);
            samples(row, 1) = centers(cluster_index, 1) + noise(rng);
        }
    }
    return samples;
}

} // namespace

int main() {
    const Eigen::MatrixXd X = make_hierarchical_clusters();

    //! [example_agglomerative_clustering]
    Skigen::AgglomerativeClustering<double> model(
        /*n_clusters=*/3, /*linkage=*/"ward", /*metric=*/"euclidean");
    const Eigen::VectorXi labels = model.fit_predict(X);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Agglomerative labels: " << labels.transpose() << "\n";
    std::cout << "Merge tree shape:     " << model.children().rows() << " x "
              << model.children().cols() << "\n";
    std::cout << "First merge:          " << model.children().row(0) << "\n";
    //! [example_agglomerative_clustering]

    return 0;
}