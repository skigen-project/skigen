// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

#include <Skigen/GaussianProcess>

#include <Eigen/Core>
#include <iostream>

int main() {
    //! [example_gaussian_process_regressor]
    Eigen::MatrixXd X(6, 1);
    X << -2.0, -1.2, -0.4, 0.4, 1.2, 2.0;
    Eigen::VectorXd y(6);
    y << 0.15, -0.55, -0.38, 0.42, 0.95, 0.55;

    using GPR = Skigen::GaussianProcessRegressor<double>;
    GPR model(
        GPR::Kernel::RBF,
        /*alpha=*/1e-6,
        /*length_scale=*/0.8,
        /*constant_value=*/1.0,
        /*noise_level=*/1.0,
        /*nu=*/1.5,
        /*rational_quadratic_alpha=*/1.0,
        /*periodicity=*/1.0,
        /*sigma_0=*/1.0,
        /*normalize_y=*/true);
    model.fit(X, y);

    Eigen::MatrixXd X_test(5, 1);
    X_test << -1.8, -0.8, 0.0, 0.8, 1.8;
    const Eigen::VectorXd mean = model.predict(X_test);
    const Eigen::VectorXd std = model.predict_std(X_test);
    //! [example_gaussian_process_regressor]

    std::cout << "GaussianProcessRegressor predictions\n";
    for (Eigen::Index i = 0; i < X_test.rows(); ++i) {
        std::cout << "x=" << X_test(i, 0)
                  << " mean=" << mean(i)
                  << " std=" << std(i) << '\n';
    }
    std::cout << "log_marginal_likelihood="
              << model.log_marginal_likelihood_value() << '\n';
    return 0;
}
