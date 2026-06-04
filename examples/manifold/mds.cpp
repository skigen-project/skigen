// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// mds.cpp — Multidimensional Scaling (metric MDS via SMACOF).
//
// Equivalent scikit-learn snippet:
//
//   from sklearn.manifold import MDS
//   import numpy as np
//   X = np.array([[0,0],[1,0],[0,1],[1,1]], dtype=float)
//   mds = MDS(n_components=2, max_iter=300, random_state=42)
//   Y = mds.fit_transform(X)
//   print(Y, mds.stress_, mds.n_iter_)

#include <Skigen/Manifold>

#include <Eigen/Core>
#include <iomanip>
#include <iostream>

int main() {
    Eigen::MatrixXd X(4, 2);
    X << 0, 0,
         1, 0,
         0, 1,
         1, 1;

    Skigen::MDS<double> mds(/*n_components=*/2, /*max_iter=*/300,
                            /*tol=*/1e-6, /*metric=*/true, /*random_state=*/42);
    auto Y = mds.fit_transform(X);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== MDS (metric, SMACOF) ===\n";
    std::cout << "  embedding shape = " << Y.rows() << " x " << Y.cols() << "\n";
    std::cout << "  final stress    = " << mds.stress() << "\n";
    std::cout << "  iterations      = " << mds.n_iter() << "\n";
    return 0;
}
