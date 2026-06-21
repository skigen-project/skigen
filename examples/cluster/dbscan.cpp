// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

#include <Skigen/Cluster>

#include <Eigen/Core>

#include <iomanip>
#include <iostream>
#include <random>

namespace {

Eigen::MatrixXd make_density_clusters() {
    constexpr int cluster_samples = 24;
    constexpr int noise_samples = 3;
    Eigen::MatrixXd samples(cluster_samples * 2 + noise_samples, 2);

    std::mt19937 rng(7);
    std::normal_distribution<double> noise(0.0, 0.12);
    for (int sample_index = 0; sample_index < cluster_samples; ++sample_index) {
        samples(sample_index, 0) = -2.0 + noise(rng);
        samples(sample_index, 1) = noise(rng);
        samples(cluster_samples + sample_index, 0) = 2.0 + noise(rng);
        samples(cluster_samples + sample_index, 1) = noise(rng);
    }
    samples.row(cluster_samples * 2 + 0) << -4.0, 3.0;
    samples.row(cluster_samples * 2 + 1) << 0.0, -3.0;
    samples.row(cluster_samples * 2 + 2) << 4.0, 3.0;
    return samples;
}

} // namespace

int main() {
    const Eigen::MatrixXd X = make_density_clusters();

    //! [example_dbscan]
    Skigen::DBSCAN<double> dbscan(/*eps=*/0.45, /*min_samples=*/4);
    const Eigen::VectorXi labels = dbscan.fit_predict(X);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "DBSCAN labels: " << labels.transpose() << "\n";
    std::cout << "Core samples:  " << dbscan.core_sample_indices().size() << "\n";
    std::cout << "Components:    " << dbscan.components().rows() << " x "
              << dbscan.components().cols() << "\n";
    //! [example_dbscan]

    return 0;
}