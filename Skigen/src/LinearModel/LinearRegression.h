// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_LINEAR_MODEL_LINEAR_REGRESSION_H
#define SKIGEN_LINEAR_MODEL_LINEAR_REGRESSION_H

#include "../Core/Base.h"
#include "../Core/Validation.h"

#include <Eigen/Core>
#include <Eigen/QR>

namespace Skigen {

/// LinearRegression — Ordinary Least Squares.
/// Mirrors sklearn.linear_model.LinearRegression.
///
/// Solves  min_w ||y - Xw||²  via ColPivHouseholderQR.
/// When fit_intercept=true, data is centered before solving.
template <typename Scalar = double>
class LinearRegression
    : public Predictor<LinearRegression<Scalar>, Scalar> {
public:
    using Base = Predictor<LinearRegression<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;
    using typename Base::ScalarType;

    explicit LinearRegression(bool fit_intercept = true)
        : fit_intercept_(fit_intercept) {}

    // -- Accessors ----------------------------------------------------------

    [[nodiscard]] bool fit_intercept() const noexcept { return fit_intercept_; }

    [[nodiscard]] const RowVectorType& coef() const {
        this->check_is_fitted(); return coef_;
    }
    [[nodiscard]] Scalar intercept() const {
        this->check_is_fitted(); return intercept_;
    }
    [[nodiscard]] IndexType rank() const {
        this->check_is_fitted(); return rank_;
    }

    // -- Implementation (called by CRTP base) --------------------------------

    LinearRegression& fit_impl(const Eigen::Ref<const MatrixType>& X,
                               const Eigen::Ref<const VectorType>& y) {
        internal::check_non_empty(X);
        internal::check_consistent_length(X, y);

        this->n_features_in_ = X.cols();

        if (fit_intercept_) {
            x_offset_ = X.colwise().mean();
            const Scalar y_offset = y.mean();

            MatrixType X_c = X.rowwise() - x_offset_;
            VectorType y_c = y.array() - y_offset;

            auto qr = X_c.colPivHouseholderQr();
            coef_ = qr.solve(y_c).transpose();
            rank_ = qr.rank();

            intercept_ = y_offset - x_offset_.dot(coef_.transpose());
        } else {
            auto qr = X.colPivHouseholderQr();
            coef_ = qr.solve(y).transpose();
            rank_ = qr.rank();
            intercept_ = Scalar{0};
        }

        this->fitted_ = true;
        return *this;
    }

    [[nodiscard]] VectorType predict_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        return (X * coef_.transpose()).array() + intercept_;
    }

    [[nodiscard]] ScalarType score_impl(
        const Eigen::Ref<const MatrixType>& X,
        const Eigen::Ref<const VectorType>& y) const {
        internal::check_consistent_length(X, y);
        VectorType y_pred = predict_impl(X);
        Scalar ss_res = (y - y_pred).squaredNorm();
        Scalar ss_tot = (y.array() - y.mean()).matrix().squaredNorm();
        if (ss_tot == Scalar{0}) return Scalar{0};
        return Scalar{1} - ss_res / ss_tot;
    }

private:
    bool fit_intercept_;

    RowVectorType coef_;
    Scalar intercept_ = Scalar{0};
    RowVectorType x_offset_;
    IndexType rank_ = 0;
};

} // namespace Skigen

#endif // SKIGEN_LINEAR_MODEL_LINEAR_REGRESSION_H
