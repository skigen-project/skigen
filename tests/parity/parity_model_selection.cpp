// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors
//
// Behavioural parity for GridSearchCV / RandomizedSearchCV over a Ridge alpha
// grid: the selected best estimator's held-out R^2 must match sklearn's
// within a band.

#include <Skigen/Dense>

#include "parity_common.h"

namespace {
using namespace skigen_parity;

Skigen::ParameterGrid alpha_grid(const std::string& e) {
    Eigen::VectorXd alphas = load_vector(e, "alphas.csv");
    std::vector<Skigen::ParameterValue> values;
    for (Eigen::Index i = 0; i < alphas.size(); ++i)
        values.emplace_back(alphas(i));
    return Skigen::ParameterGrid(
        Skigen::ParameterGrid::Grid{{"alpha", values}});
}

void grid_search_cv() {
    const std::string e = "grid_search_cv";
    Eigen::MatrixXd Xtr = load_matrix(e, "X_train.csv");
    Eigen::VectorXd ytr = load_vector(e, "y_train.csv");
    Eigen::MatrixXd Xte = load_matrix(e, "X_test.csv");
    Eigen::VectorXd yte = load_vector(e, "y_test.csv");
    const double ref = load_vector(e, "test_r2.csv")(0);

    Skigen::Ridge<double> est;
    Skigen::GridSearchCV<Skigen::Ridge<double>> gs(est, alpha_grid(e), 5);
    gs.fit(Xtr, ytr);
    expect_score(r2_score(gs.predict(Xte), yte), ref, 0.10, e + ".test_r2");
}

void randomized_search_cv() {
    const std::string e = "randomized_search_cv";
    Eigen::MatrixXd Xtr = load_matrix(e, "X_train.csv");
    Eigen::VectorXd ytr = load_vector(e, "y_train.csv");
    Eigen::MatrixXd Xte = load_matrix(e, "X_test.csv");
    Eigen::VectorXd yte = load_vector(e, "y_test.csv");
    const double ref = load_vector(e, "test_r2.csv")(0);

    Skigen::Ridge<double> est;
    Skigen::RandomizedSearchCV<Skigen::Ridge<double>> rs(
        est, alpha_grid(e), /*n_iter=*/6, /*cv=*/5, /*refit=*/true,
        std::optional<uint64_t>{0});
    rs.fit(Xtr, ytr);
    expect_score(r2_score(rs.predict(Xte), yte), ref, 0.10, e + ".test_r2");
}
}  // namespace

void parity_model_selection() {
    run("GridSearchCV", grid_search_cv);
    run("RandomizedSearchCV", randomized_search_cv);
}
