// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

// anomaly.cpp — EllipticEnvelope and IsolationForest on dense outlier data.
//
// Equivalent scikit-learn snippet:
//
//   from sklearn.covariance import EllipticEnvelope
//   from sklearn.ensemble import IsolationForest
//   envelope = EllipticEnvelope(contamination=0.2).fit(X)
//   forest = IsolationForest(contamination=0.2, random_state=7).fit(X)

#include <Skigen/Anomaly>

#include <Eigen/Core>
#include <iomanip>
#include <iostream>

int main() {
    Eigen::MatrixXd X(10, 2);
    X << -0.6, -0.4,
         -0.3,  0.2,
          0.0, -0.1,
          0.4,  0.3,
          0.6, -0.2,
         -0.5,  0.5,
          0.2,  0.6,
          0.5,  0.1,
          5.0,  5.5,
         -5.0,  4.5;

    std::cout << std::fixed << std::setprecision(4);

    //! [example_elliptic_envelope]
    Skigen::EllipticEnvelope<double> envelope(/*contamination=*/0.2);
    envelope.fit(X);
    Eigen::VectorXi envelope_labels = envelope.predict(X);

    std::cout << "=== EllipticEnvelope ===\n";
    std::cout << "offset: " << envelope.offset() << "\n";
    std::cout << "labels: " << envelope_labels.transpose() << "\n\n";
    //! [example_elliptic_envelope]

    //! [example_isolation_forest]
    Skigen::IsolationForest<double> forest(
        /*n_estimators=*/64,
        /*max_samples=*/8,
        /*contamination=*/0.2,
        /*max_depth=*/-1,
        /*random_state=*/7);
    forest.fit(X);
    Eigen::VectorXi forest_labels = forest.predict(X);

    std::cout << "=== IsolationForest ===\n";
    std::cout << "offset: " << forest.offset() << "\n";
    std::cout << "labels: " << forest_labels.transpose() << "\n";
    //! [example_isolation_forest]

    return 0;
}
