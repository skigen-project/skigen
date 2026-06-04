// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// gradient_boosting_classifier.cpp — GradientBoostingClassifier on a binary
// 2-D classification problem.
//
// Equivalent scikit-learn snippet:
//
//   from sklearn.ensemble import GradientBoostingClassifier
//   import numpy as np
//   rng = np.random.default_rng(11)
//   X = np.vstack([rng.normal(loc=-1, scale=0.5, size=(100, 2)),
//                  rng.normal(loc= 1, scale=0.5, size=(100, 2))])
//   y = np.concatenate([np.zeros(100), np.ones(100)]).astype(int)
//   gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
//                                   max_depth=3, random_state=11)
//   gb.fit(X, y)
//   print("training accuracy =", gb.score(X, y))
//   print("init_             =", gb.init_)
//   print("first/last loss   =", gb.train_score_[0], gb.train_score_[-1])

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

    using GBC = Skigen::GradientBoostingClassifier<double>;
    GBC gb(GBC::Loss::LogLoss,
           /*learning_rate=*/0.1,
           /*n_estimators=*/100,
           /*subsample=*/1.0,
           GBC::CriterionGB::FriedmanMSE,
           /*min_samples_split=*/2,
           /*min_samples_leaf=*/1,
           /*min_weight_fraction_leaf=*/0.0,
           /*max_depth=*/3,
           /*min_impurity_decrease=*/0.0,
           std::optional<uint64_t>(11));
    gb.fit(X, y);

    auto pred = gb.predict(X);
    int correct = 0;
    for (int i = 0; i < n; ++i) if (pred(i) == y(i)) ++correct;
    const double acc = static_cast<double>(correct) / n;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== GradientBoostingClassifier ===\n";
    std::cout << "  n_estimators       = " << gb.estimators().size() << "\n";
    std::cout << "  init_              = " << gb.init() << "\n";
    std::cout << "  training accuracy  = " << acc << "\n";
    std::cout << "  first stage loss   = " << gb.train_score()(0) << "\n";
    std::cout << "  final stage loss   = "
              << gb.train_score()(gb.train_score().size() - 1) << "\n";
    return 0;
}
