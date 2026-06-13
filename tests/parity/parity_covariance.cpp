// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#include <Skigen/Dense>

#include "parity_common.h"

namespace {
using namespace skigen_parity;

void empirical_covariance() {
    const std::string e = "empirical_covariance";
    Eigen::MatrixXd X = load_matrix(e, "X.csv");
    Skigen::EmpiricalCovariance<double> cov;
    cov.fit(X);
    expect_allclose(cov.covariance(), load_matrix(e, "covariance.csv"), 1e-10,
                    "EmpiricalCovariance.covariance_");
    expect_allclose(cov.location().transpose(),
                    load_vector(e, "location.csv"), 1e-10,
                    "EmpiricalCovariance.location_");
}

void ledoit_wolf() {
    const std::string e = "ledoit_wolf";
    Eigen::MatrixXd X = load_matrix(e, "X.csv");
    Skigen::LedoitWolf<double> cov;
    cov.fit(X);
    expect_allclose(cov.covariance(), load_matrix(e, "covariance.csv"), 1e-9,
                    "LedoitWolf.covariance_");
    expect_near(cov.shrinkage(), load_vector(e, "shrinkage.csv")(0), 1e-9,
                "LedoitWolf.shrinkage_");
}

void oas() {
    const std::string e = "oas";
    Eigen::MatrixXd X = load_matrix(e, "X.csv");
    Skigen::OAS<double> cov;
    cov.fit(X);
    expect_allclose(cov.covariance(), load_matrix(e, "covariance.csv"), 1e-9,
                    "OAS.covariance_");
    expect_near(cov.shrinkage(), load_vector(e, "shrinkage.csv")(0), 1e-9,
                "OAS.shrinkage_");
}
}  // namespace

void parity_covariance() {
    run("EmpiricalCovariance", empirical_covariance);
    run("LedoitWolf", ledoit_wolf);
    run("OAS", oas);
}
