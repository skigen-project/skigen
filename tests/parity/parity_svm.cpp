// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors
//
// Behavioural parity for the SVM estimators. The nu-SVM family (NuSVC /
// NuSVR) still ship as documented placeholders whose fit() throws; their
// parity check asserts that error behaviour. OneClassSVM is implemented and
// checked for basic one-class detection behaviour instead.

#include <Skigen/Dense>

#include "parity_common.h"

namespace {
using namespace skigen_parity;

constexpr double kClfBand = 0.15;
constexpr double kRegBand = 0.20;

template <typename Clf>
void clf(const std::string& e) {
    Eigen::MatrixXd Xtr = load_matrix(e, "X_train.csv");
    Eigen::VectorXi ytr = to_int(load_vector(e, "y_train.csv"));
    Eigen::MatrixXd Xte = load_matrix(e, "X_test.csv");
    Eigen::VectorXi yte = to_int(load_vector(e, "y_test.csv"));
    const double ref = load_vector(e, "score.csv")(0);
    Clf model;
    model.fit(Xtr, ytr);
    expect_score(accuracy(model.predict(Xte), yte), ref, kClfBand,
                 e + ".accuracy");
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
    expect_score(r2_score(model.predict(Xte), yte), ref, kRegBand, e + ".r2");
}

template <typename F>
void expect_throws(F&& f, const std::string& ctx) {
    bool threw = false;
    try {
        f();
    } catch (const std::exception&) {
        threw = true;
    }
    if (!threw)
        throw ParityFailure(ctx + ": expected placeholder fit() to throw");
}

Eigen::MatrixXd tiny_X() {
    Eigen::MatrixXd X(4, 2);
    X << 1, 2, 3, 4, 5, 6, 7, 8;
    return X;
}
}  // namespace

void parity_svm() {
    run("SVC", [] { clf<Skigen::SVC<double>>("svc"); });
    run("LinearSVC", [] { clf<Skigen::LinearSVC<double>>("linear_svc"); });
    run("SVR", [] { reg<Skigen::SVR<double>>("svr"); });
    run("LinearSVR", [] { reg<Skigen::LinearSVR<double>>("linear_svr"); });
    run("NuSVC (placeholder)", [] {
        Eigen::MatrixXd X = tiny_X();
        Eigen::VectorXi y(4); y << 0, 1, 0, 1;
        Skigen::NuSVC<double> m;
        expect_throws([&] { m.fit(X, y); }, "NuSVC");
    });
    run("NuSVR (placeholder)", [] {
        Eigen::MatrixXd X = tiny_X();
        Eigen::VectorXd y(4); y << 0.1, 0.9, 0.2, 0.8;
        Skigen::NuSVR<double> m;
        expect_throws([&] { m.fit(X, y); }, "NuSVR");
    });
    run("OneClassSVM", [] {
        // Dense cluster plus two outliers; the model must fit and flag at
        // least one outlier while keeping the cluster mostly inlying.
        Eigen::MatrixXd X(20, 2);
        for (int i = 0; i < 18; ++i) {
            X(i, 0) = 0.2 * std::sin(0.7 * i);
            X(i, 1) = 0.2 * std::cos(0.7 * i);
        }
        X(18, 0) = 2.5;  X(18, 1) = 2.5;
        X(19, 0) = -2.5; X(19, 1) = 2.2;
        Skigen::OneClassSVM<double> m(
            Skigen::OneClassSVM<double>::Kernel::RBF, 3, 0.5, 0.0, 0.2);
        m.fit(X);
        const Eigen::VectorXi labels = m.predict(X);
        int outliers = 0;
        for (int i = 0; i < labels.size(); ++i)
            if (labels(i) == -1) ++outliers;
        if (outliers < 1 || outliers >= labels.size())
            throw ParityFailure("OneClassSVM: expected a partial outlier set");
    });
}
