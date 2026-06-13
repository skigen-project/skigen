// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors
//
// Behavioural parity for CalibratedClassifierCV (binary, sigmoid) wrapping a
// GaussianNB base, matching scikit-learn's configuration.

#include <Skigen/Dense>

#include "parity_common.h"

namespace {
using namespace skigen_parity;

void calibrated_classifier_cv() {
    const std::string e = "calibrated_classifier_cv";
    Eigen::MatrixXd Xtr = load_matrix(e, "X_train.csv");
    Eigen::VectorXi ytr = to_int(load_vector(e, "y_train.csv"));
    Eigen::MatrixXd Xte = load_matrix(e, "X_test.csv");
    Eigen::VectorXi yte = to_int(load_vector(e, "y_test.csv"));
    const double ref_acc = load_vector(e, "accuracy.csv")(0);
    const double ref_brier = load_vector(e, "brier.csv")(0);

    Skigen::GaussianNB<double> nb;
    Skigen::CalibratedClassifierCV<Skigen::GaussianNB<double>, double> cc(
        nb, Skigen::CalibrationMethod::Sigmoid, 5);
    cc.fit(Xtr, ytr);

    expect_score(accuracy(cc.predict(Xte), yte), ref_acc, 0.12,
                 e + ".accuracy");

    // Brier score: lower is better, so Skigen must not exceed sklearn's by
    // more than the band.
    Eigen::MatrixXd P = cc.predict_proba(Xte);
    double brier = 0.0;
    for (Eigen::Index i = 0; i < yte.size(); ++i) {
        const double d = P(i, 1) - static_cast<double>(yte(i));
        brier += d * d;
    }
    brier /= static_cast<double>(yte.size());
    if (brier > ref_brier + 0.10) {
        throw ParityFailure(e + ".brier: " + std::to_string(brier) +
                            " > sklearn " + std::to_string(ref_brier) +
                            " + 0.10");
    }
}
}  // namespace

void parity_calibration() {
    run("CalibratedClassifierCV", calibrated_classifier_cv);
}
