// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_LINEAR_MODEL_RIDGE_H
#define SKIGEN_LINEAR_MODEL_RIDGE_H

#include "../Core/Base.h"
#include "../Core/Validation.h"

#include <Eigen/Core>
#include <Eigen/Cholesky>

namespace Skigen {

/// Ridge — L2-regularized least squares.
/// Mirrors sklearn.linear_model.Ridge (cholesky solver).
///
/// Solves  min_w ||y - Xw||² + α||w||²
/// Via the normal equation: (X^T X + αI) w = X^T y
template <typename Scalar = double>
class Ridge
    : public Predictor<Ridge<Scalar>, Scalar> {
public:
    using Base = Predictor<Ridge<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;
    using typename Base::ScalarType;

    explicit Ridge(Scalar alpha = Scalar{1}, bool fit_intercept = true)
        : alpha_(alpha), fit_intercept_(fit_intercept) {}

    // -- Accessors ----------------------------------------------------------

    [[nodiscard]] Scalar alpha() const noexcept { return alpha_; }
    [[nodiscard]] bool fit_intercept() const noexcept { return fit_intercept_; }

    [[nodiscard]] const RowVectorType& coef() const {
        this->check_is_fitted(); return coef_;
    }
    [[nodiscard]] Scalar intercept() const {
        this->check_is_fitted(); return intercept_;
    }

    // -- Implementation (called by CRTP base) --------------------------------

    Ridge& fit_impl(const Eigen::Ref<const MatrixType>& X,
                    const Eigen::Ref<const VectorType>& y) {
        internal::check_non_empty(X);
        internal::check_consistent_length(X, y);

        this->n_features_in_ = X.cols();

        MatrixType X_c;
        VectorType y_c;
        Scalar y_offset{0};

        if (fit_intercept_) {
            x_offset_ = X.colwise().mean();
            y_offset = y.mean();
            X_c = X.rowwise() - x_offset_;
            y_c = y.array() - y_offset;
        } else {
            X_c = X;
            y_c = y;
        }

        // (X^T X + αI) w = X^T y — Cholesky solver
        MatrixType XtX = X_c.transpose() * X_c;
        XtX.diagonal().array() += alpha_;

        VectorType Xty = X_c.transpose() * y_c;

        Eigen::LLT<MatrixType> llt(XtX);
        VectorType w = llt.solve(Xty);

        coef_ = w.transpose();

        if (fit_intercept_) {
            intercept_ = y_offset - x_offset_.dot(w);
        } else {
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
    Scalar alpha_;
    bool fit_intercept_;

    RowVectorType coef_;
    Scalar intercept_ = Scalar{0};
    RowVectorType x_offset_;
};

} // namespace Skigen

#endif // SKIGEN_LINEAR_MODEL_RIDGE_H
