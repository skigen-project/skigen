// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors
//
// Behavioural parity for the ensemble estimators: Skigen must reach a
// held-out score within a band of scikit-learn's on the same split.

#include <Skigen/Dense>

#include "parity_common.h"

namespace {
using namespace skigen_parity;

constexpr double kClfBand = 0.15;  // accuracy
constexpr double kRegBand = 0.20;  // R^2

template <typename Clf>
void clf(const std::string& e) {
    Eigen::MatrixXd Xtr = load_matrix(e, "X_train.csv");
    Eigen::VectorXi ytr = to_int(load_vector(e, "y_train.csv"));
    Eigen::MatrixXd Xte = load_matrix(e, "X_test.csv");
    Eigen::VectorXi yte = to_int(load_vector(e, "y_test.csv"));
    const double ref = load_vector(e, "score.csv")(0);

    Clf model;
    model.fit(Xtr, ytr);
    const double got = accuracy(model.predict(Xte), yte);
    expect_score(got, ref, kClfBand, e + ".accuracy");
}

template <typename Reg>
void reg(const std::string& e) {
    Eigen::MatrixXd Xtr = load_matrix(e, "X_train.csv");
    Eigen::VectorXd ytr = load_vector(e, "y_train.csv");
    Eigen::MatrixXd Xte = load_matrix(e, "X_test.csv");
    Eigen::VectorXd yte = load_vector(e, "y_test.csv");
    const double ref = load_vector(e, "score.csv")(0);

    Reg model;
    model.fit(Xtr, ytr);
    const double got = r2_score(model.predict(Xte), yte);
    expect_score(got, ref, kRegBand, e + ".r2");
}
}  // namespace

void parity_ensemble() {
    run("RandomForestClassifier",
        [] { clf<Skigen::RandomForestClassifier<double>>(
                 "random_forest_classifier"); });
    run("RandomForestRegressor",
        [] { reg<Skigen::RandomForestRegressor<double>>(
                 "random_forest_regressor"); });
    run("GradientBoostingClassifier",
        [] { clf<Skigen::GradientBoostingClassifier<double>>(
                 "gradient_boosting_classifier"); });
    run("GradientBoostingRegressor",
        [] { reg<Skigen::GradientBoostingRegressor<double>>(
                 "gradient_boosting_regressor"); });
    run("HistGradientBoostingClassifier",
        [] { clf<Skigen::HistGradientBoostingClassifier<double>>(
                 "hist_gradient_boosting_classifier"); });
    run("HistGradientBoostingRegressor",
        [] { reg<Skigen::HistGradientBoostingRegressor<double>>(
                 "hist_gradient_boosting_regressor"); });
}
