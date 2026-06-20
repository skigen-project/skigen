// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// svc.cpp — kernel SVC on a 2-cluster Gaussian dataset.
//
// Equivalent scikit-learn snippet:
//
//   from sklearn.svm import SVC
//   import numpy as np
//   rng = np.random.default_rng(11)
//   X = np.vstack([rng.normal(loc=-1, scale=0.5, size=(100, 2)),
//                  rng.normal(loc= 1, scale=0.5, size=(100, 2))])
//   y = np.concatenate([np.zeros(100), np.ones(100)]).astype(int)
//   clf = SVC(C=1.0, kernel="rbf", gamma=0.5)
//   clf.fit(X, y)
//   print("training accuracy =", clf.score(X, y))
//   print("n_support_        =", clf.n_support_)

#include <Skigen/SVM>

#include <Eigen/Core>
#include <iomanip>
#include <iostream>
#include <random>

#ifdef SKIGEN_EXAMPLE_WITH_PLOT
#include <skigen/plot/figure.h>
#endif

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
    constexpr int n = 200;
    std::mt19937_64 rng(11);
    std::normal_distribution<double> nz(0.0, 0.5);

    Eigen::MatrixXd X(n, 2);
    Eigen::VectorXi y(n);
    for (int i = 0; i < n; ++i) {
        const double cls = (i < n / 2) ? -1.0 : 1.0;
        X(i, 0) = cls + nz(rng);
        X(i, 1) = cls + nz(rng);
        y(i)    = (cls > 0) ? 1 : 0;
    }

    //! [example_svc]
    using K = Skigen::SVC<double>::Kernel;
    Skigen::SVC<double> clf(/*C=*/1.0, K::RBF, /*degree=*/3, /*gamma=*/0.5);
    clf.fit(X, y);

    auto pred = clf.predict(X);
    int correct = 0;
    for (int i = 0; i < n; ++i) if (pred(i) == y(i)) ++correct;
    const double acc = static_cast<double>(correct) / n;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== SVC (RBF kernel) ===\n";
    std::cout << "  training accuracy = " << acc << "\n";
    std::cout << "  n_support         = " << clf.n_support() << "\n";
    //! [example_svc]

#ifdef SKIGEN_EXAMPLE_WITH_PLOT
    //! [example_svc_decision_regions_plot]
    constexpr int grid_side = 90;
    Eigen::MatrixXd grid(grid_side * grid_side, 2);
    const double x_min = X.col(0).minCoeff() - 0.6;
    const double x_max = X.col(0).maxCoeff() + 0.6;
    const double y_min = X.col(1).minCoeff() - 0.6;
    const double y_max = X.col(1).maxCoeff() + 0.6;

    int row = 0;
    for (int iy = 0; iy < grid_side; ++iy) {
        const double gy = y_min + (y_max - y_min) * iy / (grid_side - 1);
        for (int ix = 0; ix < grid_side; ++ix) {
            const double gx = x_min + (x_max - x_min) * ix / (grid_side - 1);
            grid(row, 0) = gx;
            grid(row, 1) = gy;
            ++row;
        }
    }

    const Eigen::VectorXi grid_labels = clf.predict(grid);
    Eigen::MatrixXd support_points(static_cast<Eigen::Index>(clf.support().size()), 2);
    Eigen::VectorXi support_labels(static_cast<Eigen::Index>(clf.support().size()));
    for (Eigen::Index i = 0; i < support_points.rows(); ++i) {
        const Eigen::Index sample = clf.support()[static_cast<std::size_t>(i)];
        support_points.row(i) = X.row(sample);
        support_labels(i) = y(sample);
    }

    Skigen::Plot::Figure fig;
    fig.title("SVC Decision Regions")
       .caption("RBF-kernel Skigen::SVC decision regions with support vectors highlighted")
       .xlabel("feature 1")
       .ylabel("feature 2")
       .scatter(grid, grid_labels, {.pointSize = 2.0f, .opacity = 0.16f})
       .scatter(X, y, {.pointSize = 7.0f, .opacity = 0.92f})
       .scatter(support_points, support_labels, {.pointSize = 15.0f, .hollow = true});

    return argc > 1 ? (fig.saveThemed(argv[1]) ? 0 : 1) : fig.show();
    //! [example_svc_decision_regions_plot]
#else
    return 0;
#endif
}
