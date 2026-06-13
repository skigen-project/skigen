// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#include <Skigen/Dense>

#include "parity_common.h"

namespace {
using namespace skigen_parity;

void isotonic() {
    const std::string e = "isotonic";
    Eigen::VectorXd x = load_vector(e, "x.csv");
    Eigen::VectorXd y = load_vector(e, "y.csv");
    Eigen::MatrixXd X = x;  // n x 1

    Skigen::IsotonicRegression<double> iso(std::nullopt, std::nullopt,
                                           Skigen::IsotonicIncreasing::True,
                                           Skigen::OutOfBounds::Clip);
    iso.fit(X, y);
    Eigen::VectorXd yhat = iso.predict(X);
    expect_allclose(yhat, load_vector(e, "yhat.csv"), 1e-9,
                    "IsotonicRegression.predict");
}
}  // namespace

void parity_isotonic() { run("IsotonicRegression", isotonic); }
