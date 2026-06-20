// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

// nearest_neighbors.cpp — brute-force nearest-neighbor queries.
//
// Equivalent scikit-learn snippet:
//
//   from sklearn.neighbors import NearestNeighbors
//   import numpy as np
//   X = np.array([[0, 0], [1, 0], [0, 2], [3, 0]], dtype=float)
//   nn = NearestNeighbors(n_neighbors=2, radius=1.0).fit(X)
//   print(nn.kneighbors([[0.2, 0.0]], return_distance=True))

#include <Skigen/Neighbors>

#include <Eigen/Core>
#include <iostream>

int main() {
    Eigen::MatrixXd X(4, 2);
    X << 0.0, 0.0,
         1.0, 0.0,
         0.0, 2.0,
         3.0, 0.0;
    Eigen::MatrixXd Q(1, 2);
    Q << 0.2, 0.0;

    //! [example_nearest_neighbors]
    Skigen::NearestNeighbors<double> nn(/*n_neighbors=*/2, /*radius=*/1.0);
    nn.fit(X);

    Eigen::MatrixXi indices = nn.kneighbors(Q);
    Eigen::MatrixXd distances = nn.kneighbors_distances(Q);
    auto radius_indices = nn.radius_neighbors(Q);

    std::cout << "=== NearestNeighbors ===\n";
    std::cout << "indices: " << indices << "\n";
    std::cout << "distances: " << distances << "\n";
    std::cout << "radius count: " << radius_indices[0].size() << "\n";
    //! [example_nearest_neighbors]

    return 0;
}
