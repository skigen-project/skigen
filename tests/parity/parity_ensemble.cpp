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

// Factory variants: the caller supplies a configured (non-default) estimator.
template <typename Clf, typename Factory>
void clf_with(const std::string& e, Factory make, double band = kClfBand) {
    Eigen::MatrixXd Xtr = load_matrix(e, "X_train.csv");
    Eigen::VectorXi ytr = to_int(load_vector(e, "y_train.csv"));
    Eigen::MatrixXd Xte = load_matrix(e, "X_test.csv");
    Eigen::VectorXi yte = to_int(load_vector(e, "y_test.csv"));
    const double ref = load_vector(e, "score.csv")(0);

    Clf model = make();
    model.fit(Xtr, ytr);
    const double got = accuracy(model.predict(Xte), yte);
    expect_score(got, ref, band, e + ".accuracy");
}

template <typename Reg, typename Factory>
void reg_with(const std::string& e, Factory make, double band = kRegBand) {
    Eigen::MatrixXd Xtr = load_matrix(e, "X_train.csv");
    Eigen::VectorXd ytr = load_vector(e, "y_train.csv");
    Eigen::MatrixXd Xte = load_matrix(e, "X_test.csv");
    Eigen::VectorXd yte = load_vector(e, "y_test.csv");
    const double ref = load_vector(e, "score.csv")(0);

    Reg model = make();
    model.fit(Xtr, ytr);
    const double got = r2_score(model.predict(Xte), yte);
    expect_score(got, ref, band, e + ".r2");
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

    // --- Non-default parameter branches (§4.8 / §4.11) ---
    using GBR = Skigen::GradientBoostingRegressor<double>;
    run("GradientBoostingRegressor[absolute_error]", [] {
        reg_with<GBR>("gradient_boosting_regressor_absolute_error",
                      [] { return GBR(GBR::Loss::AbsoluteError, 0.1, 100); });
    });
    run("GradientBoostingRegressor[huber]", [] {
        reg_with<GBR>("gradient_boosting_regressor_huber",
                      [] { return GBR(GBR::Loss::Huber, 0.1, 100); });
    });
    run("GradientBoostingRegressor[quantile]", [] {
        // sklearn quantile alpha=0.5 is the 12th ctor arg.
        reg_with<GBR>("gradient_boosting_regressor_quantile", [] {
            return GBR(GBR::Loss::Quantile, 0.1, 100, 1.0,
                       GBR::CriterionGB::FriedmanMSE, 2, 1, 0.0, 3, 0.0,
                       std::nullopt, /*alpha=*/0.5);
        });
    });
    run("GradientBoostingRegressor[subsample]", [] {
        reg_with<GBR>("gradient_boosting_regressor_subsample", [] {
            return GBR(GBR::Loss::SquaredError, 0.1, 100, /*subsample=*/0.7,
                       GBR::CriterionGB::FriedmanMSE, 2, 1, 0.0, 3, 0.0,
                       std::optional<uint64_t>(0));
        });
    });
    run("GradientBoostingClassifier[multiclass]", [] {
        clf<Skigen::GradientBoostingClassifier<double>>(
            "gradient_boosting_classifier_multiclass");
    });
    run("HistGradientBoostingClassifier[multiclass]", [] {
        clf<Skigen::HistGradientBoostingClassifier<double>>(
            "hist_gradient_boosting_classifier_multiclass");
    });
    using HGBR = Skigen::HistGradientBoostingRegressor<double>;
    run("HistGradientBoostingRegressor[leafwise]", [] {
        reg_with<HGBR>("hist_gradient_boosting_regressor_leafwise", [] {
            return HGBR(HGBR::Loss::SquaredError, 0.1, 100,
                        /*max_leaf_nodes=*/8, std::nullopt, 20,
                        /*l2=*/1.0);
        });
    });
}
