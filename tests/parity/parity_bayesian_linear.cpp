// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#include <Skigen/Dense>

#include "parity_common.h"

namespace {
using namespace skigen_parity;

void bayesian_ridge() {
    const std::string e = "bayesian_ridge";
    Eigen::MatrixXd X = load_matrix(e, "X.csv");
    Eigen::VectorXd y = load_vector(e, "y.csv");
    Skigen::BayesianRidge<double> br;
    br.fit(X, y);
    expect_allclose(br.coef(), load_matrix(e, "coef.csv").row(0), 1e-5,
                    "BayesianRidge.coef_");
    expect_near(br.intercept(), load_vector(e, "intercept.csv")(0), 1e-5,
                "BayesianRidge.intercept_");
    expect_near(br.alpha(), load_vector(e, "alpha.csv")(0), 1e-3,
                "BayesianRidge.alpha_");
    expect_near(br.lambda_(), load_vector(e, "lambda_.csv")(0), 1e-3,
                "BayesianRidge.lambda_");
    expect_allclose(br.predict(X), load_vector(e, "pred.csv"), 1e-5,
                    "BayesianRidge.predict");
}

void ard_regression() {
    const std::string e = "ard_regression";
    Eigen::MatrixXd X = load_matrix(e, "X.csv");
    Eigen::VectorXd y = load_vector(e, "y.csv");
    Skigen::ARDRegression<double> ard;
    ard.fit(X, y);
    // ARD's per-feature pruning can leave small coefficient differences; the
    // predictive mean is the stable, identifiable quantity to compare.
    expect_allclose(ard.predict(X), load_vector(e, "pred.csv"), 1e-3,
                    "ARDRegression.predict");
    expect_allclose(ard.coef(), load_matrix(e, "coef.csv").row(0), 1e-2,
                    "ARDRegression.coef_");
}
}  // namespace

void parity_bayesian_linear() {
    run("BayesianRidge", bayesian_ridge);
    run("ARDRegression", ard_regression);
}
