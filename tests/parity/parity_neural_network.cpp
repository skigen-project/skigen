// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors
//
// Behavioural parity for the MLP estimators (fixed architecture + seed).

#include <Skigen/Dense>

#include "parity_common.h"

namespace {
using namespace skigen_parity;

// Match the architecture used by the generator: one hidden layer of 32 ReLU
// units, Adam, 500 iterations, seed 0.
template <typename MLP>
MLP make_mlp() {
    return MLP({32}, Skigen::MLPActivation::ReLU, Skigen::MLPSolver::Adam,
               1e-4, 1e-3, 500, 1e-4, 0, std::optional<uint64_t>{0});
}

void mlp_classifier() {
    const std::string e = "mlp_classifier";
    Eigen::MatrixXd Xtr = load_matrix(e, "X_train.csv");
    Eigen::VectorXi ytr = to_int(load_vector(e, "y_train.csv"));
    Eigen::MatrixXd Xte = load_matrix(e, "X_test.csv");
    Eigen::VectorXi yte = to_int(load_vector(e, "y_test.csv"));
    const double ref = load_vector(e, "score.csv")(0);

    auto mlp = make_mlp<Skigen::MLPClassifier<double>>();
    mlp.fit(Xtr, ytr);
    expect_score(accuracy(mlp.predict(Xte), yte), ref, 0.20, e + ".accuracy");
}

void mlp_regressor() {
    const std::string e = "mlp_regressor";
    Eigen::MatrixXd Xtr = load_matrix(e, "X_train.csv");
    Eigen::VectorXd ytr = load_vector(e, "y_train.csv");
    Eigen::MatrixXd Xte = load_matrix(e, "X_test.csv");
    Eigen::VectorXd yte = load_vector(e, "y_test.csv");
    const double ref = load_vector(e, "score.csv")(0);

    auto mlp = make_mlp<Skigen::MLPRegressor<double>>();
    mlp.fit(Xtr, ytr);
    expect_score(r2_score(mlp.predict(Xte), yte), ref, 0.25, e + ".r2");
}
}  // namespace

void parity_neural_network() {
    run("MLPClassifier", mlp_classifier);
    run("MLPRegressor", mlp_regressor);
}
