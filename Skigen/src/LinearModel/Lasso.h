// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_LINEAR_MODEL_LASSO_H
#define SKIGEN_LINEAR_MODEL_LASSO_H

#include "../Core/Base.h"
#include "../Core/Validation.h"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>

namespace Skigen {

/// @defgroup Algo_Lasso Lasso Regression
/// @ingroup LinearModels
/// @brief L1-regularized least squares (Lasso) via coordinate descent.
/// @see https://skigen-project.github.io/docs/guide/lasso for algorithm intuition.
/// @{

/// @brief Linear Model trained with L1 prior as regularizer (aka the Lasso).
///
/// The optimization objective for Lasso is:
///
/// @f[
///   \frac{1}{2n_{\mathrm{samples}}} \|y - Xw\|_2^2
///   + \alpha \|w\|_1
/// @f]
///
/// Technically the Lasso model is optimizing the same objective function
/// as the ElasticNet with `l1_ratio = 1.0` (no L2 penalty).
///
/// The L1 penalty induces sparsity, driving some coefficients exactly
/// to zero.
///
/// Mirrors
/// [sklearn.linear_model.Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html).
///
/// Read more in the @ref guide_lasso "User Guide".
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `alpha` | `Scalar` | `1` | Constant that multiplies the L1 term, controlling regularization strength. `alpha = 0` is equivalent to ordinary least squares (use LinearRegression instead). |
/// | `fit_intercept` | `bool` | `true` | Whether the intercept should be estimated. If `false`, the data is assumed to be already centered. |
/// | `max_iter` | `int` | `1000` | Maximum number of coordinate descent iterations. |
/// | `tol` | `Scalar` | `1e-4` | The tolerance for the optimization: iterations stop when the maximum coordinate update is smaller than `tol`. |
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type | Description |
/// |----------|------|-------------|
/// | `coef()` | `RowVectorType` | Parameter vector @f$w@f$ of shape (1 × n_features). |
/// | `intercept()` | `Scalar` | Independent term in the decision function. |
///
/// ### See also
///
/// - Skigen::LinearRegression — Ordinary least squares without regularization.
/// - Skigen::Ridge — L2-only regularization.
/// - Skigen::ElasticNet — Combined L1 + L2 regularization.
///
/// ### Notes
///
/// The algorithm used to fit the model is coordinate descent with
/// soft-thresholding. To avoid unnecessary memory duplication the `X`
/// argument should ideally be column-major (Eigen's default).
///
/// @note **scikit-learn parity gaps:** The following sklearn constructor
///   parameters are not yet supported: `precompute`, `copy_X`, `warm_start`,
///   `positive`, `random_state`, `selection`.
///   The following sklearn fitted attributes are not yet exposed:
///   `n_iter_`, `dual_gap_`, `sparse_coef_`, `n_features_in_`,
///   `feature_names_in_`.
///   `sample_weight` in `fit()` is not yet supported.
///
/// ### Examples
///
/// @snippet lasso.cpp example_lasso
template <typename Scalar = double>
class Lasso
    : public Predictor<Lasso<Scalar>, Scalar> {
public:
    using Base = Predictor<Lasso<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;

    /// @brief Construct a Lasso estimator.
    ///
    /// @param alpha Constant that multiplies the L1 term (`Scalar`, default `1`).
    ///   Controls regularization strength. `alpha = 0` is equivalent to
    ///   ordinary least squares (use LinearRegression instead for numerical
    ///   stability).
    /// @param fit_intercept Whether the intercept should be estimated (`bool`, default `true`).
    ///   If `false`, the data is assumed to be already centered.
    /// @param max_iter The maximum number of iterations (`int`, default `1000`).
    /// @param tol The tolerance for the optimization (`Scalar`, default `1e-4`):
    ///   if the maximum coordinate update is smaller than `tol`, the solver stops.
    explicit Lasso(Scalar alpha = Scalar{1}, bool fit_intercept = true,
                   int max_iter = 1000, Scalar tol = Scalar{1e-4})
        : alpha_(alpha), fit_intercept_(fit_intercept),
          max_iter_(max_iter), tol_(tol) {}

    /// @brief Parameter vector @f$w@f$ (1 × n_features).
    ///
    /// Estimated coefficients for the linear regression problem.
    /// @return Read-only reference to the coefficient row-vector.
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] const RowVectorType& coef() const {
        this->check_is_fitted(); return coef_;
    }
    /// @brief Independent term in the decision function.
    ///
    /// Set to `0` if `fit_intercept = false`.
    /// @return The intercept value.
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] Scalar intercept() const {
        this->check_is_fitted(); return intercept_;
    }

    /// @brief Fit the Lasso model via coordinate descent.
    ///
    /// Centers the data when `fit_intercept` is `true`, then runs
    /// coordinate descent with soft-thresholding until convergence
    /// or `max_iter` iterations.
    ///
    /// @param X Design matrix of shape (n_samples, n_features).
    /// @param y Target vector of shape (n_samples,).
    ///   Will be cast to `Scalar` if necessary.
    /// @return Reference to the fitted estimator (`*this`).
    ///
    /// @note **sklearn parity gap:** `sample_weight` and `check_input`
    ///   parameters are not yet supported.
    Lasso& fit_impl(const Eigen::Ref<const MatrixType>& X,
                    const Eigen::Ref<const VectorType>& y) {
        internal::check_non_empty(X);
        this->n_features_in_ = X.cols();
        const Eigen::Index n = X.rows();
        const Eigen::Index p = X.cols();

        // Center data if fit_intercept
        RowVectorType X_mean = RowVectorType::Zero(p);
        Scalar y_mean{0};

        if (fit_intercept_) {
            X_mean = X.colwise().mean();
            y_mean = y.mean();
        }

        MatrixType X_c;
        VectorType y_c;

        if (fit_intercept_) {
            X_c = (X.rowwise() - X_mean).eval();
            y_c = (y.array() - y_mean).matrix().eval();
        } else {
            X_c = X;
            y_c = y;
        }

        // Precompute column norms squared
        VectorType col_norms_sq(p);
        for (Eigen::Index j = 0; j < p; ++j) {
            col_norms_sq(j) = X_c.col(j).squaredNorm();
        }

        // Coordinate descent
        coef_ = RowVectorType::Zero(p);
        VectorType residual = y_c;

        for (int iter = 0; iter < max_iter_; ++iter) {
            Scalar max_change{0};

            for (Eigen::Index j = 0; j < p; ++j) {
                if (col_norms_sq(j) < std::numeric_limits<Scalar>::epsilon()) continue;

                Scalar old_w = coef_(j);

                // Partial residual
                Scalar rho = X_c.col(j).dot(residual) + col_norms_sq(j) * old_w;

                // Soft-thresholding
                Scalar na = static_cast<Scalar>(n) * alpha_;
                coef_(j) = soft_threshold(rho, na) / col_norms_sq(j);

                // Update residual
                Scalar delta = coef_(j) - old_w;
                if (delta != Scalar{0}) {
                    residual -= delta * X_c.col(j);
                }

                max_change = std::max(max_change, std::abs(delta));
            }

            if (max_change < tol_) break;
        }

        intercept_ = fit_intercept_
            ? y_mean - (X_mean.array() * coef_.array()).sum()
            : Scalar{0};

        this->fitted_ = true;
        return *this;
    }

    /// @brief Predict using the linear model.
    ///
    /// Computes @f$ \hat{y} = X w + b @f$ where @f$w@f$ and @f$b@f$ are
    /// the fitted coefficients and intercept.
    ///
    /// @param X Sample matrix of shape (n_samples, n_features).
    /// @return Predicted values of shape (n_samples,).
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] VectorType predict_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        return (X * coef_.transpose()).array() + intercept_;
    }

    /// @brief Return the @f$R^2@f$ coefficient of determination on test data.
    ///
    /// @f$ R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2} @f$.
    /// Best possible score is 1.0; it can be negative if the model is
    /// arbitrarily worse than predicting the mean.
    ///
    /// @param X Test samples of shape (n_samples, n_features).
    /// @param y True values of shape (n_samples,).
    /// @return @f$R^2@f$ score.
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] Scalar score_impl(const Eigen::Ref<const MatrixType>& X,
                                    const Eigen::Ref<const VectorType>& y) const {
        VectorType y_pred = predict_impl(X);
        Scalar ss_res = (y - y_pred).squaredNorm();
        Scalar ss_tot = (y.array() - y.mean()).matrix().squaredNorm();
        if (ss_tot == Scalar{0}) return Scalar{0};
        return Scalar{1} - ss_res / ss_tot;
    }

private:
    Scalar alpha_;
    bool fit_intercept_;
    int max_iter_;
    Scalar tol_;

    RowVectorType coef_;
    Scalar intercept_{0};

    static Scalar soft_threshold(Scalar x, Scalar lambda) {
        if (x > lambda) return x - lambda;
        if (x < -lambda) return x + lambda;
        return Scalar{0};
    }
};

/// @}

} // namespace Skigen

#endif // SKIGEN_LINEAR_MODEL_LASSO_H
