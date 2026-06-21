// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

#include <Skigen/GaussianProcess>

#include <Eigen/Core>
#include <iostream>

int main() {
    //! [example_gaussian_process_classifier]
    Eigen::MatrixXd X(8, 2);
    X << -2.0, -1.0,
         -1.5, -0.6,
         -1.0, -1.2,
         -0.7, -0.3,
          0.7,  0.4,
          1.0,  1.1,
          1.4,  0.5,
          2.0,  1.2;
    Eigen::VectorXi y(8);
    y << 0, 0, 0, 0, 1, 1, 1, 1;

    using GPC = Skigen::GaussianProcessClassifier<double>;
    GPC model(
        GPC::Kernel::RBF,
        /*alpha=*/1e-6,
        /*length_scale=*/1.0,
        /*constant_value=*/1.0);
    model.fit(X, y);

    Eigen::MatrixXd X_test(3, 2);
    X_test << -1.2, -0.8,
               0.0,  0.0,
               1.2,  0.9;
    const Eigen::VectorXi labels = model.predict(X_test);
    const Eigen::MatrixXd probabilities = model.predict_proba(X_test);
    //! [example_gaussian_process_classifier]

    std::cout << "GaussianProcessClassifier predictions\n";
    for (Eigen::Index i = 0; i < X_test.rows(); ++i) {
        std::cout << "label=" << labels(i)
                  << " p0=" << probabilities(i, 0)
                  << " p1=" << probabilities(i, 1) << '\n';
    }
    std::cout << "n_iter=" << model.n_iter() << '\n';
    return 0;
}
