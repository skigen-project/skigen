// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#include <Skigen/Dense>

#include "parity_common.h"

namespace {
using namespace skigen_parity;

void factor_analysis() {
    const std::string e = "factor_analysis";
    Eigen::MatrixXd X = load_matrix(e, "X.csv");
    Skigen::FactorAnalysis<double> fa(2, 1000, 1e-3);
    fa.fit(X);

    // Loadings are identifiable only up to rotation, so compare the
    // rotation-invariant noise variance and model-implied covariance.
    expect_allclose(fa.noise_variance(),
                    load_vector(e, "noise_variance.csv"), 5e-2,
                    "FactorAnalysis.noise_variance_");

    Eigen::MatrixXd W = fa.components();  // p x k
    Eigen::MatrixXd implied =
        W * W.transpose() +
        fa.noise_variance().asDiagonal().toDenseMatrix();
    expect_allclose(implied, load_matrix(e, "implied_cov.csv"), 5e-2,
                    "FactorAnalysis.implied_cov");
}
}  // namespace

void parity_decomposition() { run("FactorAnalysis", factor_analysis); }
