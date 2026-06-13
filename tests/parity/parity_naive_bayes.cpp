// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors
//
// Parity checks for the Naive Bayes estimators against scikit-learn.

#include <Skigen/Dense>

#include "parity_common.h"

namespace {

using namespace skigen_parity;

void gaussian_nb() {
    const std::string e = "gaussian_nb";
    Eigen::MatrixXd X = load_matrix(e, "X.csv");
    Eigen::VectorXi y = to_int(load_vector(e, "y.csv"));

    Skigen::GaussianNB<double> nb;
    nb.fit(X, y);

    expect_allclose(nb.theta(), load_matrix(e, "theta.csv"), 1e-9,
                    "GaussianNB.theta_");
    expect_allclose(nb.var(), load_matrix(e, "var.csv"), 1e-9,
                    "GaussianNB.var_");
    expect_allclose(nb.class_prior(), load_vector(e, "class_prior.csv"), 1e-12,
                    "GaussianNB.class_prior_");
    expect_allclose(nb.predict_proba(X), load_matrix(e, "proba.csv"), 1e-9,
                    "GaussianNB.predict_proba");
}

void multinomial_nb() {
    const std::string e = "multinomial_nb";
    Eigen::MatrixXd X = load_matrix(e, "X.csv");
    Eigen::VectorXi y = to_int(load_vector(e, "y.csv"));

    Skigen::MultinomialNB<double> nb;
    nb.fit(X, y);

    expect_allclose(nb.feature_log_prob(),
                    load_matrix(e, "feature_log_prob.csv"), 1e-12,
                    "MultinomialNB.feature_log_prob_");
    expect_allclose(nb.class_log_prior(),
                    load_vector(e, "class_log_prior.csv"), 1e-12,
                    "MultinomialNB.class_log_prior_");
    expect_allclose(nb.predict_proba(X), load_matrix(e, "proba.csv"), 1e-12,
                    "MultinomialNB.predict_proba");
}

void bernoulli_nb() {
    const std::string e = "bernoulli_nb";
    Eigen::MatrixXd X = load_matrix(e, "X.csv");
    Eigen::VectorXi y = to_int(load_vector(e, "y.csv"));

    Skigen::BernoulliNB<double> nb;
    nb.fit(X, y);

    expect_allclose(nb.feature_log_prob(),
                    load_matrix(e, "feature_log_prob.csv"), 1e-12,
                    "BernoulliNB.feature_log_prob_");
    expect_allclose(nb.class_log_prior(),
                    load_vector(e, "class_log_prior.csv"), 1e-12,
                    "BernoulliNB.class_log_prior_");
    expect_allclose(nb.predict_proba(X), load_matrix(e, "proba.csv"), 1e-12,
                    "BernoulliNB.predict_proba");
}

}  // namespace

void parity_naive_bayes() {
    run("GaussianNB", gaussian_nb);
    run("MultinomialNB", multinomial_nb);
    run("BernoulliNB", bernoulli_nb);
}
