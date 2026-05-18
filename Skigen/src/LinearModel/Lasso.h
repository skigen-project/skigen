// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_LINEAR_MODEL_LASSO_H
#define SKIGEN_LINEAR_MODEL_LASSO_H

#include "../Core/Base.h"
#include "../Core/Validation.h"

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <algorithm>
#include <cmath>
#include <limits>

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
/// ### Limitations relative to scikit-learn
///
/// The following scikit-learn constructor
///   parameters are not honoured: `precompute`, `copy_X`, `warm_start`,
///   `positive`, `random_state`, `selection`.
///   The following sklearn fitted attributes are not exposed:
///   `n_iter_`, `dual_gap_`, `sparse_coef_`, `n_features_in_`,
///   `feature_names_in_`.
///   `sample_weight` in `fit()` is not honoured.
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

    // Make the dense base-class fit overload visible alongside the
    // sparse fit overload added below.
    using Base::fit;

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

    SKIGEN_PARAMS(
        (alpha,         alpha_,         double),
        (fit_intercept, fit_intercept_, bool),
        (max_iter,      max_iter_,      int),
        (tol,           tol_,           double))

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
    /// ### Limitations relative to scikit-learn
///
/// `sample_weight` and `check_input`
    ///   parameters are not honoured.
    Lasso& fit_impl(const Eigen::Ref<const MatrixType>& X,
                    const Eigen::Ref<const VectorType>& y) {
        internal::check_non_empty(X);
        this->n_features_in_ = X.cols();
        // Reset multi-target view so it's lazily re-synthesised.
        coef_matrix_.resize(0, 0);
        intercept_vector_.resize(0);
        n_targets_ = 1;
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

    // -- Sparse-aware overload --------------------------------

    /// @brief Fit Lasso on a sparse design matrix without densifying X.
    ///
    /// Implements coordinate descent with implicit centring. Per-column
    /// quantities are computed directly from the CSC nonzeros; the residual
    /// is updated via sparse column-iteration. We maintain
    /// `r_raw = y_c - X*w` (using *uncentered* X but centered `y_c = y - ȳ`)
    /// and a scalar `shift = Σ_k w_k · x̄_k`. The CD update on the
    /// centered problem then reduces to
    /// @f$ \rho_j = X[:, j] \cdot r^{raw} + n\,\bar{x}_j\,\mathrm{shift}
    ///             + \|X_c[:, j]\|^2 \, w_j^{old} @f$
    /// where @f$ \|X_c[:, j]\|^2 = \|X[:, j]\|^2 - n\,\bar{x}_j^2 @f$,
    /// so the centered design is never materialised.
    ///
    /// Mirrors sklearn's `Lasso.fit` behaviour on sparse input.
    /// `sample_weight`, `precompute`, `positive`, `random_state`,
    /// `selection`, and `warm_start` are not honoured.
    template <int Options, typename StorageIndex>
    Lasso& fit(const Eigen::SparseMatrix<Scalar, Options, StorageIndex>& X,
               const Eigen::Ref<const VectorType>& y) {
        if (X.rows() == 0 || X.cols() == 0) {
            throw std::invalid_argument("Lasso.fit: empty sparse matrix.");
        }
        if (X.rows() != y.size()) {
            throw std::invalid_argument(
                "Lasso.fit: X has " + std::to_string(X.rows()) +
                " rows but y has " + std::to_string(y.size()) + ".");
        }
        this->n_features_in_ = X.cols();

        using ColSparse =
            Eigen::SparseMatrix<Scalar, Eigen::ColMajor, StorageIndex>;
        const ColSparse Xc = X;
        const Eigen::Index n = Xc.rows();
        const Eigen::Index p = Xc.cols();

        // Per-column means and squared norms from CSC nonzeros.
        RowVectorType x_mean = RowVectorType::Zero(p);
        VectorType    raw_norm_sq = VectorType::Zero(p);
        for (Eigen::Index j = 0; j < p; ++j) {
            Scalar s{0}, ss{0};
            for (typename ColSparse::InnerIterator it(Xc, j); it; ++it) {
                s  += it.value();
                ss += it.value() * it.value();
            }
            x_mean(j)      = s / static_cast<Scalar>(n);
            raw_norm_sq(j) = ss;
        }

        const Scalar y_mean = fit_intercept_ ? y.mean() : Scalar{0};

        // Centred-column squared norms via the identity
        // ||X_c[:, j]||^2 = ||X[:, j]||^2 - n * x̄_j^2 (when fit_intercept=true);
        // when fit_intercept=false the raw norm is used directly.
        VectorType col_norms_sq(p);
        for (Eigen::Index j = 0; j < p; ++j) {
            if (fit_intercept_) {
                col_norms_sq(j) = raw_norm_sq(j)
                                  - static_cast<Scalar>(n) *
                                    x_mean(j) * x_mean(j);
            } else {
                col_norms_sq(j) = raw_norm_sq(j);
            }
            // Numerical safety: clamp tiny negatives that arise from FP
            // cancellation in the implicit-centring identity.
            if (col_norms_sq(j) < Scalar{0}) col_norms_sq(j) = Scalar{0};
        }

        coef_ = RowVectorType::Zero(p);

        // r_raw = y_c - X * w (uses uncentered X but centered y_c).
        // Initial w = 0 ⇒ r_raw = y_c.
        VectorType r_raw = fit_intercept_
            ? VectorType((y.array() - y_mean).matrix())
            : VectorType(y);

        // shift = Σ_k w_k · x̄_k (initial 0). Only used when fit_intercept_.
        Scalar shift{0};

        const Scalar na = static_cast<Scalar>(n) * alpha_;

        for (int iter = 0; iter < max_iter_; ++iter) {
            Scalar max_change{0};

            for (Eigen::Index j = 0; j < p; ++j) {
                if (col_norms_sq(j) < std::numeric_limits<Scalar>::epsilon()) {
                    continue;
                }
                const Scalar old_w = coef_(j);

                // X[:, j] · r_raw via sparse column iteration.
                Scalar dot{0};
                for (typename ColSparse::InnerIterator it(Xc, j); it; ++it) {
                    dot += it.value() * r_raw(it.row());
                }

                // Add the implicit-centring correction term n*x̄_j*shift
                // when fit_intercept_ (zero otherwise — x_mean(j) is 0).
                Scalar rho = dot + col_norms_sq(j) * old_w;
                if (fit_intercept_) {
                    rho += static_cast<Scalar>(n) * x_mean(j) * shift;
                }

                const Scalar new_w = soft_threshold(rho, na) / col_norms_sq(j);
                const Scalar delta = new_w - old_w;

                if (delta != Scalar{0}) {
                    // r_raw -= delta * X[:, j]  (sparse update — only nz rows)
                    for (typename ColSparse::InnerIterator it(Xc, j);
                         it; ++it) {
                        r_raw(it.row()) -= delta * it.value();
                    }
                    if (fit_intercept_) {
                        shift += delta * x_mean(j);
                    }
                    coef_(j) = new_w;
                    if (std::abs(delta) > max_change) max_change = std::abs(delta);
                }
            }

            if (max_change < tol_) break;
        }

        intercept_ = fit_intercept_ ? y_mean - shift : Scalar{0};

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

    // -- Multi-target regression ------------------------------

    /// @brief Fit Lasso with a multi-target response matrix.
    ///
    /// Each target column is fitted independently with the same coordinate
    /// descent (`max_iter`, `tol`, soft-thresholding); the per-column
    /// design quantities (`X_mean`, `col_norms_sq`) are computed once and
    /// shared across targets.
    ///
    /// Additive: the single-target API (`coef()` / `intercept()`) keeps
    /// reflecting the first target column. Multi-target accessors:
    /// `coef_matrix()`, `intercept_vector()`, `n_targets()`,
    /// `predict_multi(X)`.
    Lasso& fit_multi(const Eigen::Ref<const MatrixType>& X,
                     const Eigen::Ref<const MatrixType>& Y) {
        internal::check_non_empty(X);
        if (X.rows() != Y.rows()) {
            throw std::invalid_argument(
                "fit_multi: X has " + std::to_string(X.rows()) +
                " rows but Y has " + std::to_string(Y.rows()) + ".");
        }
        if (Y.cols() < 1) {
            throw std::invalid_argument(
                "fit_multi: Y must have at least 1 target column.");
        }
        this->n_features_in_ = X.cols();
        const Eigen::Index n = X.rows();
        const Eigen::Index p = X.cols();
        const Eigen::Index t = Y.cols();

        // Shared design-side precomputation.
        RowVectorType X_mean = RowVectorType::Zero(p);
        RowVectorType y_offset = RowVectorType::Zero(t);
        if (fit_intercept_) {
            X_mean   = X.colwise().mean();
            y_offset = Y.colwise().mean();
        }
        MatrixType X_c = fit_intercept_
            ? MatrixType((X.rowwise() - X_mean).eval())
            : MatrixType(X);

        VectorType col_norms_sq(p);
        for (Eigen::Index j = 0; j < p; ++j) {
            col_norms_sq(j) = X_c.col(j).squaredNorm();
        }

        coef_matrix_ = MatrixType::Zero(t, p);
        intercept_vector_ = VectorType::Zero(t);

        const Scalar na = static_cast<Scalar>(n) * alpha_;

        for (Eigen::Index k = 0; k < t; ++k) {
            VectorType y_c = fit_intercept_
                ? VectorType((Y.col(k).array() - y_offset(k)).matrix())
                : VectorType(Y.col(k));

            RowVectorType w = RowVectorType::Zero(p);
            VectorType residual = y_c;

            for (int iter = 0; iter < max_iter_; ++iter) {
                Scalar max_change{0};
                for (Eigen::Index j = 0; j < p; ++j) {
                    if (col_norms_sq(j) <
                        std::numeric_limits<Scalar>::epsilon()) continue;
                    const Scalar old_w = w(j);
                    const Scalar rho = X_c.col(j).dot(residual) +
                                       col_norms_sq(j) * old_w;
                    w(j) = soft_threshold(rho, na) / col_norms_sq(j);
                    const Scalar delta = w(j) - old_w;
                    if (delta != Scalar{0}) residual -= delta * X_c.col(j);
                    if (std::abs(delta) > max_change) max_change = std::abs(delta);
                }
                if (max_change < tol_) break;
            }

            coef_matrix_.row(k) = w;
            intercept_vector_(k) = fit_intercept_
                ? y_offset(k) - (X_mean.array() * w.array()).sum()
                : Scalar{0};
        }

        // Mirror first target into single-target accessors.
        coef_      = coef_matrix_.row(0);
        intercept_ = intercept_vector_(0);
        n_targets_ = static_cast<int>(t);
        this->fitted_ = true;
        return *this;
    }

    [[nodiscard]] const MatrixType& coef_matrix() const {
        this->check_is_fitted();
        if (coef_matrix_.size() == 0) {
            coef_matrix_.resize(1, coef_.size());
            coef_matrix_.row(0) = coef_;
            intercept_vector_.resize(1);
            intercept_vector_(0) = intercept_;
            n_targets_ = 1;
        }
        return coef_matrix_;
    }
    [[nodiscard]] const VectorType& intercept_vector() const {
        this->check_is_fitted();
        if (intercept_vector_.size() == 0) {
            intercept_vector_.resize(1);
            intercept_vector_(0) = intercept_;
            coef_matrix_.resize(1, coef_.size());
            coef_matrix_.row(0) = coef_;
            n_targets_ = 1;
        }
        return intercept_vector_;
    }
    [[nodiscard]] int n_targets() const {
        this->check_is_fitted(); return n_targets_;
    }
    [[nodiscard]] MatrixType predict_multi(
        const Eigen::Ref<const MatrixType>& X) const {
        this->check_is_fitted();
        this->validate_feature_count(X);
        const MatrixType& W = coef_matrix();
        const VectorType& b = intercept_vector();
        MatrixType Y_pred = X * W.transpose();
        Y_pred.rowwise() += b.transpose();
        return Y_pred;
    }

private:
    Scalar alpha_;
    bool fit_intercept_;
    int max_iter_;
    Scalar tol_;

    RowVectorType coef_;
    Scalar intercept_{0};

    // Multi-target state (lazy 1-target view for single-target fit).
    mutable MatrixType  coef_matrix_;
    mutable VectorType  intercept_vector_;
    mutable int         n_targets_ = 0;

    static Scalar soft_threshold(Scalar x, Scalar lambda) {
        if (x > lambda) return x - lambda;
        if (x < -lambda) return x + lambda;
        return Scalar{0};
    }
};

/// @}

} // namespace Skigen

#endif // SKIGEN_LINEAR_MODEL_LASSO_H
