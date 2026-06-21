// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

#include <Skigen/Cluster>

#include <Eigen/Core>

#include <iomanip>
#include <iostream>
#include <random>

namespace {

Eigen::MatrixXd make_streaming_clusters() {
    constexpr int samples_per_cluster = 24;
    Eigen::MatrixXd samples(samples_per_cluster * 3, 2);

    std::mt19937 rng(31);
    std::normal_distribution<double> noise(0.0, 0.15);
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
    const Eigen::MatrixXd X = make_streaming_clusters();

    //! [example_birch]
    Skigen::Birch<double> birch(/*threshold=*/0.45, /*n_clusters=*/3);
    const Eigen::VectorXi labels = birch.fit_predict(X);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Birch labels:       " << labels.transpose() << "\n";
    std::cout << "Subcluster centers: " << birch.subcluster_centers().rows() << "\n";
    std::cout << "Cluster centers:\n" << birch.cluster_centers() << "\n";
    //! [example_birch]

    return 0;
}