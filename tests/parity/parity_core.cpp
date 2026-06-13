// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors
//
// Parity for the core linear and preprocessing estimators (LinearRegression,
// Ridge, Lasso, ElasticNet, KMeans, StandardScaler). Closed-form / convex
// estimators get tight tolerances; KMeans uses a behavioural inertia band
// (k-means++ seeding differs).

#include <Skigen/Dense>

#include "parity_common.h"

namespace {
using namespace skigen_parity;

void standard_scaler() {
    const std::string e = "standard_scaler";
    Eigen::MatrixXd X = load_matrix(e, "X.csv");
    Skigen::StandardScaler sc;
    sc.fit(X);
    expect_allclose(sc.mean(), load_matrix(e, "mean.csv").row(0), 1e-12,
                    "StandardScaler.mean_");
    expect_allclose(sc.scale(), load_matrix(e, "scale.csv").row(0), 1e-12,
                    "StandardScaler.scale_");
    expect_allclose(sc.transform(X), load_matrix(e, "Z.csv"), 1e-12,
                    "StandardScaler.transform");
}

template <typename Reg>
void linear(const std::string& e, Reg model, double tol) {
    Eigen::MatrixXd X = load_matrix(e, "X.csv");
    Eigen::VectorXd y = load_vector(e, "y.csv");
    model.fit(X, y);
    expect_allclose(model.coef(), load_matrix(e, "coef.csv").row(0), tol,
                    e + ".coef_");
    expect_near(model.intercept(), load_vector(e, "intercept.csv")(0), tol,
                e + ".intercept_");
    expect_allclose(model.predict(X), load_vector(e, "pred.csv"), tol,
                    e + ".predict");
}

void kmeans() {
    const std::string e = "kmeans";
    Eigen::MatrixXd X = load_matrix(e, "X.csv");
    const double ref = load_vector(e, "inertia.csv")(0);
    Skigen::KMeans<double> km(3);
    km.fit(X);
    // Both minimise the same objective; Skigen must not be materially worse.
    if (km.inertia() > ref * 1.15 + 1e-9) {
        throw ParityFailure(e + ".inertia: " + std::to_string(km.inertia()) +
                            " > sklearn " + std::to_string(ref) + " * 1.15");
    }
}
}  // namespace

void parity_core() {
    run("StandardScaler", standard_scaler);
    run("LinearRegression",
        [] { linear("linear_regression", Skigen::LinearRegression<double>(),
                    1e-9); });
    run("Ridge",
        [] { linear("ridge", Skigen::Ridge<double>(1.0, true), 1e-7); });
    run("Lasso",
        [] { linear("lasso", Skigen::Lasso<double>(0.1, true), 1e-5); });
    run("ElasticNet",
        [] { linear("elastic_net", Skigen::ElasticNet<double>(0.1, 0.5, true),
                    1e-5); });
    run("KMeans", kmeans);
}
