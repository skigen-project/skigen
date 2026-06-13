// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#include <Skigen/Dense>

#include "parity_common.h"

namespace {
using namespace skigen_parity;

void variance_threshold() {
    const std::string e = "variance_threshold";
    Eigen::MatrixXd X = load_matrix(e, "X.csv");
    Skigen::VarianceThreshold<double> vt(0.0);
    vt.fit(X);
    expect_allclose(vt.variances(), load_matrix(e, "variances.csv").row(0),
                    1e-10, "VarianceThreshold.variances_");
}

template <typename ScoreFn, typename YVec>
void univariate(const std::string& e, ScoreFn fn, const YVec& y) {
    Eigen::MatrixXd X = load_matrix(e, "X.csv");
    Skigen::SelectKBest<double, ScoreFn> sel(fn, 2);
    sel.fit(X, y);
    expect_allclose(sel.scores(), load_matrix(e, "scores.csv").row(0), 1e-7,
                    e + ".scores_");
    expect_allclose(sel.pvalues(), load_matrix(e, "pvalues.csv").row(0), 1e-7,
                    e + ".pvalues_");
}

void f_classif() {
    Eigen::VectorXi y = to_int(load_vector("f_classif", "y.csv"));
    univariate("f_classif", Skigen::feature_selection::FClassif<double>{}, y);
}

void f_regression() {
    Eigen::VectorXd y = load_vector("f_regression", "y.csv");
    univariate("f_regression", Skigen::feature_selection::FRegression<double>{},
               y);
}

void chi2() {
    Eigen::VectorXi y = to_int(load_vector("chi2", "y.csv"));
    univariate("chi2", Skigen::feature_selection::Chi2<double>{}, y);
}
}  // namespace

void parity_feature_selection() {
    run("VarianceThreshold", variance_threshold);
    run("f_classif", f_classif);
    run("f_regression", f_regression);
    run("chi2", chi2);
}
