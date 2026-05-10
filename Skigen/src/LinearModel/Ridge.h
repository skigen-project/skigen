// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_LINEAR_MODEL_RIDGE_H
#define SKIGEN_LINEAR_MODEL_RIDGE_H

#include "../Core/Base.h"
#include "../Core/Validation.h"

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/SparseCore>

namespace Skigen {

/// @defgroup Algo_Ridge Ridge Regression
/// @ingroup LinearModels
/// @brief L2-regularized least squares (Ridge regression).
/// @see https://skigen-project.github.io/docs/guide/ridge for algorithm intuition.
/// @{

/// @brief Linear least squares with L2 regularization.
///
/// Minimizes the objective function:
///
/// @f[
///   \|y - Xw\|_2^2 + \alpha \|w\|_2^2
/// @f]
///
/// This model solves a regression model where the loss function is the
/// linear least squares function and regularization is given by the
/// L2-norm. Also known as Ridge Regression or Tikhonov regularization.
///
/// Mirrors
/// [sklearn.linear_model.Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html).
///
/// Read more in the @ref guide_ridge "User Guide".
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `alpha` | `Scalar` | `1` | Constant that multiplies the L2 term, controlling regularization strength. `alpha = 0` is equivalent to ordinary least squares (use LinearRegression instead). |
/// | `fit_intercept` | `bool` | `true` | Whether the intercept should be estimated. If `false`, the data is assumed to be already centered. |
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
/// - Skigen::Lasso — L1-only regularization (coordinate descent).
/// - Skigen::ElasticNet — Combined L1 + L2 regularization.
///
/// ### Notes
///
/// Uses the Cholesky-based normal equation
/// @f$(X^\top X + \alpha I)w = X^\top y@f$.
///
/// Regularization improves the conditioning of the problem and reduces
/// the variance of the estimates. Larger values specify stronger
/// regularization.
///
/// @note **scikit-learn parity gaps:** The following sklearn constructor
///   parameters are not yet supported: `copy_X`, `max_iter`, `tol`,
///   `solver` (only Cholesky is implemented), `positive`, `random_state`.
///   The following sklearn fitted attributes are not yet exposed:
///   `n_iter_`, `n_features_in_`, `feature_names_in_`, `solver_`.
///   `sample_weight` in `fit()` is not yet supported.
///
/// ### Examples
///
/// @snippet ridge.cpp example_ridge
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

    // Make the dense base-class fit overload visible alongside the
    // sparse fit overload added below.
    using Base::fit;

    /// @brief Construct a Ridge estimator.
    ///
    /// @param alpha Constant that multiplies the L2 term (`Scalar`, default `1`).
    ///   Controls regularization strength. `alpha = 0` is equivalent to
    ///   ordinary least squares (use LinearRegression instead for numerical
    ///   stability).
    /// @param fit_intercept Whether the intercept should be estimated (`bool`, default `true`).
    ///   If `false`, the data is assumed to be already centered.
    explicit Ridge(Scalar alpha = Scalar{1}, bool fit_intercept = true)
        : alpha_(alpha), fit_intercept_(fit_intercept) {}

    // -- Accessors ----------------------------------------------------------

    /// @brief Regularization strength.
    [[nodiscard]] Scalar alpha() const noexcept { return alpha_; }
    /// @brief Whether an intercept is fitted.
    [[nodiscard]] bool fit_intercept() const noexcept { return fit_intercept_; }

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

    // -- Implementation (called by CRTP base) --------------------------------

    /// @brief Fit the Ridge model via Cholesky decomposition.
    ///
    /// Centers the data when `fit_intercept` is `true`, then solves
    /// the regularized normal equation via Cholesky factorization.
    ///
    /// @param X Design matrix of shape (n_samples, n_features).
    /// @param y Target vector of shape (n_samples,).
    ///   Will be cast to `Scalar` if necessary.
    /// @return Reference to the fitted estimator (`*this`).
    ///
    /// @note **sklearn parity gap:** `sample_weight` parameter is
    ///   not yet supported.
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

    // -- Sparse-aware overload (v1.1.0 §3.2) --------------------------------

    /// @brief Fit Ridge on a sparse design matrix without densifying X.
    ///
    /// Solves the regularised normal equation
    /// @f$ (X_c^\top X_c + \alpha I) w = X_c^\top y_c @f$ where the
    /// centring is applied implicitly via the identities
    ///
    /// @f[
    ///   X_c^\top X_c = X^\top X - n\, \bar{x}^\top \bar{x},\qquad
    ///   X_c^\top y_c = X^\top y - n\, \bar{y}\, \bar{x}^\top
    /// @f]
    ///
    /// so that `X` itself is never centered (which would densify it).
    /// `X^\top X` (`p \times p`) and `X^\top y` are computed directly on
    /// the sparse representation; only the small Gram matrix is
    /// materialised dense, then factored with Cholesky.
    ///
    /// Mirrors sklearn's `Ridge` behaviour on sparse input. `sample_weight`,
    /// `solver`, `tol`, `max_iter`, `positive` are documented parity gaps.
    template <int Options, typename StorageIndex>
    Ridge& fit(const Eigen::SparseMatrix<Scalar, Options, StorageIndex>& X,
               const Eigen::Ref<const VectorType>& y) {
        if (X.rows() == 0 || X.cols() == 0) {
            throw std::invalid_argument("Ridge.fit: empty sparse matrix.");
        }
        if (X.rows() != y.size()) {
            throw std::invalid_argument(
                "Ridge.fit: X has " + std::to_string(X.rows()) +
                " rows but y has " + std::to_string(y.size()) + ".");
        }
        this->n_features_in_ = X.cols();

        using ColSparse =
            Eigen::SparseMatrix<Scalar, Eigen::ColMajor, StorageIndex>;
        const ColSparse Xc = X;
        const Eigen::Index n = Xc.rows();
        const Eigen::Index p = Xc.cols();

        // Per-column means computed directly from the CSC nonzeros.
        RowVectorType x_mean = RowVectorType::Zero(p);
        if (fit_intercept_) {
            for (Eigen::Index j = 0; j < p; ++j) {
                Scalar s{0};
                for (typename ColSparse::InnerIterator it(Xc, j); it; ++it) {
                    s += it.value();
                }
                x_mean(j) = s / static_cast<Scalar>(n);
            }
        }
        const Scalar y_mean = fit_intercept_ ? y.mean() : Scalar{0};

        // X^T X (sparse * sparse densified into a small p×p dense matrix).
        MatrixType XtX = MatrixType(Xc.transpose() * Xc);
        VectorType Xty = Xc.transpose() * y;

        if (fit_intercept_) {
            // Implicit centring: X_c^T X_c = X^T X - n * x_mean^T x_mean.
            XtX.noalias() -= static_cast<Scalar>(n) *
                             (x_mean.transpose() * x_mean);
            // X_c^T y_c = X^T y - n * y_mean * x_mean^T.
            Xty.noalias() -= static_cast<Scalar>(n) * y_mean *
                             x_mean.transpose();
        }

        XtX.diagonal().array() += alpha_;

        Eigen::LLT<MatrixType> llt(XtX);
        VectorType w = llt.solve(Xty);

        coef_ = w.transpose();
        if (fit_intercept_) {
            intercept_ = y_mean - x_mean.dot(w);
            x_offset_ = x_mean;
        } else {
            intercept_ = Scalar{0};
        }

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

/// @}

} // namespace Skigen

#endif // SKIGEN_LINEAR_MODEL_RIDGE_H
