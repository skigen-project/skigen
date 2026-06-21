// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// hist_gradient_boosting_classifier.cpp — quantile-binned gradient
// boosting binary classifier on a 2-cluster Gaussian dataset.
//
// Equivalent scikit-learn snippet:
//
//   from sklearn.ensemble import HistGradientBoostingClassifier
//   import numpy as np
//   rng = np.random.default_rng(11)
//   X = np.vstack([rng.normal(loc=-1, scale=0.5, size=(100, 2)),
//                  rng.normal(loc= 1, scale=0.5, size=(100, 2))])
//   y = np.concatenate([np.zeros(100), np.ones(100)]).astype(int)
//   hgb = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1,
//                                        max_bins=64, random_state=11)
//   hgb.fit(X, y)
//   print("training accuracy =", hgb.score(X, y))

#include <Skigen/Ensemble>

#include <Eigen/Core>
#include <iomanip>
#include <iostream>
#include <random>

int main() {
    constexpr int n = 200;
    std::mt19937_64 rng(11);
    std::normal_distribution<double> ns(0.0, 0.5);

    Eigen::MatrixXd X(n, 2);
    Eigen::VectorXi y(n);
    for (int i = 0; i < n; ++i) {
        const double cls = (i < n / 2) ? -1.0 : 1.0;
        X(i, 0) = cls + ns(rng);
        X(i, 1) = cls + ns(rng);
        y(i)    = (cls > 0) ? 1 : 0;
    }

    using HGBC = Skigen::HistGradientBoostingClassifier<double>;
    HGBC hgb(HGBC::Loss::LogLoss, /*lr=*/0.1, /*max_iter=*/100,
             /*max_leaf_nodes=*/31, /*max_depth=*/std::nullopt,
             /*min_samples_leaf=*/2, /*l2=*/0.0, /*max_bins=*/64,
             /*monotonic_cst=*/std::nullopt, /*early_stopping=*/false,
             /*validation_fraction=*/0.1, /*n_iter_no_change=*/10,
             /*tol=*/1e-7, std::optional<uint64_t>(11));
    hgb.fit(X, y);

    auto pred = hgb.predict(X);
    int correct = 0;
    for (int i = 0; i < n; ++i) if (pred(i) == y(i)) ++correct;
    const double acc = static_cast<double>(correct) / n;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== HistGradientBoostingClassifier ===\n";
    std::cout << "  n_iter            = " << hgb.n_iter() << "\n";
    std::cout << "  init_             = " << hgb.init()(0) << "\n";
    std::cout << "  classes           = ["
              << hgb.classes()(0) << ", " << hgb.classes()(1) << "]\n";
    std::cout << "  training accuracy = " << acc << "\n";
    std::cout << "  first stage loss  = " << hgb.train_score()(0) << "\n";
    std::cout << "  final stage loss  = "
              << hgb.train_score()(hgb.train_score().size() - 1) << "\n";
    return 0;
}
