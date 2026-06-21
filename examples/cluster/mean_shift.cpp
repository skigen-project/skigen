// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

#include <Skigen/Cluster>

#include <Eigen/Core>

#include <iomanip>
#include <iostream>
#include <random>

namespace {

Eigen::MatrixXd make_modes() {
    constexpr int samples_per_mode = 20;
    Eigen::MatrixXd samples(samples_per_mode * 3, 2);

    std::mt19937 rng(21);
    std::normal_distribution<double> noise(0.0, 0.16);
    const Eigen::Matrix<double, 3, 2> centers = (Eigen::Matrix<double, 3, 2>() <<
        -3.0, 0.0,
         3.0, 0.0,
         0.0, 3.5).finished();

    for (int mode_index = 0; mode_index < 3; ++mode_index) {
        for (int sample_index = 0; sample_index < samples_per_mode; ++sample_index) {
            const int row = mode_index * samples_per_mode + sample_index;
            samples(row, 0) = centers(mode_index, 0) + noise(rng);
            samples(row, 1) = centers(mode_index, 1) + noise(rng);
        }
    }
    return samples;
}

} // namespace

int main() {
    const Eigen::MatrixXd X = make_modes();

    //! [example_mean_shift]
    Skigen::MeanShift<double> model(/*bandwidth=*/0.8, /*max_iter=*/100);
    const Eigen::VectorXi labels = model.fit_predict(X);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "MeanShift labels:  " << labels.transpose() << "\n";
    std::cout << "Cluster centers:\n" << model.cluster_centers() << "\n";
    std::cout << "Iterations:        " << model.n_iter() << "\n";
    //! [example_mean_shift]

    return 0;
}