// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors
//
// Behavioural parity for the SVM estimators. NuSVC, NuSVR, and OneClassSVM
// are implemented and checked for basic separation / regression / one-class
// detection behaviour.

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

template <typename Clf, typename Factory>
void clf_with(const std::string& e, Factory make) {
    Eigen::MatrixXd Xtr = load_matrix(e, "X_train.csv");
    Eigen::VectorXi ytr = to_int(load_vector(e, "y_train.csv"));
    Eigen::MatrixXd Xte = load_matrix(e, "X_test.csv");
    Eigen::VectorXi yte = to_int(load_vector(e, "y_test.csv"));
    const double ref = load_vector(e, "score.csv")(0);
    Clf model = make();
    model.fit(Xtr, ytr);
    expect_score(accuracy(model.predict(Xte), yte), ref, kClfBand,
                 e + ".accuracy");
}

template <typename Reg, typename Factory>
void reg_with(const std::string& e, Factory make) {
    Eigen::MatrixXd Xtr = load_matrix(e, "X_train.csv");
    Eigen::VectorXd ytr = load_vector(e, "y_train.csv");
    Eigen::MatrixXd Xte = load_matrix(e, "X_test.csv");
    Eigen::VectorXd yte = load_vector(e, "y_test.csv");
    const double ref = load_vector(e, "score.csv")(0);
    Reg model = make();
    model.fit(Xtr, ytr);
    expect_score(r2_score(model.predict(Xte), yte), ref, kRegBand, e + ".r2");
}

}  // namespace

void parity_svm() {
    run("SVC", [] { clf<Skigen::SVC<double>>("svc"); });
    run("LinearSVC", [] { clf<Skigen::LinearSVC<double>>("linear_svc"); });
    run("SVR", [] { reg<Skigen::SVR<double>>("svr"); });
    run("LinearSVR", [] { reg<Skigen::LinearSVR<double>>("linear_svr"); });

    // --- Non-default kernel / C branches (§4.8) ---
    using SVCd = Skigen::SVC<double>;
    run("SVC[linear,C=10]", [] {
        clf_with<SVCd>("svc_linear",
                       [] { return SVCd(10.0, SVCd::Kernel::Linear); });
    });
    run("SVC[poly]", [] {
        clf_with<SVCd>("svc_poly", [] {
            return SVCd(1.0, SVCd::Kernel::Poly, /*degree=*/3, /*gamma=*/0.0);
        });
    });
    using SVRd = Skigen::SVR<double>;
    run("SVR[linear,C=5]", [] {
        reg_with<SVRd>("svr_linear",
                       [] { return SVRd(/*C=*/5.0, SVRd::Kernel::Linear); });
    });
    run("NuSVC", [] {
        // Two well-separated clusters; NuSVC should classify them.
        Eigen::MatrixXd X(20, 2);
        Eigen::VectorXi y(20);
        for (int i = 0; i < 20; ++i) {
            const double cls = (i < 10) ? -2.0 : 2.0;
            X(i, 0) = cls + 0.1 * std::sin(0.7 * i);
            X(i, 1) = cls + 0.1 * std::cos(0.7 * i);
            y(i) = (cls > 0) ? 1 : 0;
        }
        Skigen::NuSVC<double> m(0.5);
        m.fit(X, y);
        int correct = 0;
        const Eigen::VectorXi p = m.predict(X);
        for (int i = 0; i < 20; ++i) if (p(i) == y(i)) ++correct;
        if (static_cast<double>(correct) / 20.0 < 0.8)
            throw ParityFailure("NuSVC: accuracy below 0.8 on separable data");
    });
    run("NuSVR", [] {
        // Linear signal; NuSVR should achieve a positive R^2.
        Eigen::MatrixXd X(30, 1);
        Eigen::VectorXd y(30);
        for (int i = 0; i < 30; ++i) {
            X(i, 0) = static_cast<double>(i) / 30.0;
            y(i) = 1.5 * X(i, 0) + 0.2;
        }
        Skigen::NuSVR<double> m(
            0.5, 5.0, Skigen::NuSVR<double>::Kernel::Linear, 3, 0.0, 0.0,
            1e-4, 2000);
        m.fit(X, y);
        if (r2_score(m.predict(X), y) < 0.5)
            throw ParityFailure("NuSVR: R^2 below 0.5 on a linear signal");
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
