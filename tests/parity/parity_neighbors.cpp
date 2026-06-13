// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#include <Skigen/Dense>

#include "parity_common.h"

namespace {
using namespace skigen_parity;

void local_outlier_factor() {
    const std::string e = "local_outlier_factor";
    Eigen::MatrixXd X = load_matrix(e, "X.csv");
    Skigen::LocalOutlierFactor<double> lof(20);
    lof.fit(X);
    expect_allclose(lof.negative_outlier_factor(),
                    load_vector(e, "negative_outlier_factor.csv"), 1e-6,
                    "LocalOutlierFactor.negative_outlier_factor_");
}

void kneighbors_classifier() {
    const std::string e = "kneighbors_classifier";
    Eigen::MatrixXd X = load_matrix(e, "X.csv");
    Eigen::VectorXi y = to_int(load_vector(e, "y.csv"));
    Eigen::MatrixXd Xt = load_matrix(e, "X_test.csv");
    Skigen::KNeighborsClassifier<double> knn(5);
    knn.fit(X, y);
    Eigen::VectorXi pred = knn.predict(Xt);
    Eigen::VectorXi ref = to_int(load_vector(e, "pred.csv"));
    for (Eigen::Index i = 0; i < pred.size(); ++i)
        expect_near(pred(i), ref(i), 0, "KNeighborsClassifier.predict");
}

void kneighbors_regressor() {
    const std::string e = "kneighbors_regressor";
    Eigen::MatrixXd X = load_matrix(e, "X.csv");
    Eigen::VectorXd y = load_vector(e, "y.csv");
    Eigen::MatrixXd Xt = load_matrix(e, "X_test.csv");
    Skigen::KNeighborsRegressor<double> knn(5);
    knn.fit(X, y);
    expect_allclose(knn.predict(Xt), load_vector(e, "pred.csv"), 1e-9,
                    "KNeighborsRegressor.predict");
}
}  // namespace

void parity_neighbors() {
    run("LocalOutlierFactor", local_outlier_factor);
    run("KNeighborsClassifier", kneighbors_classifier);
    run("KNeighborsRegressor", kneighbors_regressor);
}
