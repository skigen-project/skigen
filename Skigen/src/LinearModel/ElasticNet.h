// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_LINEAR_MODEL_ELASTIC_NET_H
#define SKIGEN_LINEAR_MODEL_ELASTIC_NET_H

#include "../Core/Base.h"
#include "../Core/Validation.h"

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <algorithm>
#include <cmath>
#include <limits>

namespace Skigen {

/// @defgroup Algo_ElasticNet ElasticNet Regression
/// @ingroup LinearModels
/// @brief Linear regression with combined L1 and L2 regularization.
/// @see https://skigen-project.github.io/docs/guide/elastic-net for algorithm intuition.
/// @{

/// @brief Linear regression with combined L1 and L2 priors as regularizer.
///
/// Minimizes the objective function:
///
/// @f[
///   \frac{1}{2n_{\mathrm{samples}}} \|y - Xw\|_2^2
///   + \alpha \cdot \texttt{l1\_ratio} \cdot \|w\|_1
///   + \frac{\alpha \cdot (1 - \texttt{l1\_ratio})}{2} \|w\|_2^2
/// @f]
///
/// If you are interested in controlling the L1 and L2 penalty separately,
/// keep in mind that this is equivalent to:
///
/// @f[
///   a \|w\|_1 + \tfrac{b}{2} \|w\|_2^2
///   \quad\text{where}\quad
///   \alpha = a + b,\;
///   \texttt{l1\_ratio} = \frac{a}{a + b}
/// @f]
///
/// The parameter `l1_ratio` corresponds to `alpha` in the glmnet R package
/// while `alpha` corresponds to the `lambda` parameter in glmnet.
/// Specifically, `l1_ratio = 1` is the Lasso penalty.
///
/// Mirrors
/// [sklearn.linear_model.ElasticNet](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html).
///
/// Read more in the @ref guide_elastic_net "User Guide".
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `alpha` | `Scalar` | `1` | Constant that multiplies the penalty terms. `alpha = 0` is equivalent to ordinary least squares (use LinearRegression instead for numerical stability). |
/// | `l1_ratio` | `Scalar` | `0.5` | The ElasticNet mixing parameter: `0 <= l1_ratio <= 1`. For `l1_ratio = 0` the penalty is pure L2 (Ridge). For `l1_ratio = 1` it is pure L1 (Lasso). |
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
/// - Skigen::Lasso — L1-only regularization (coordinate descent).
///
/// ### Notes
///
/// The coordinate descent solver iterates over features sequentially.
/// To avoid unnecessary memory duplication the `X` argument should
/// ideally be column-major (Eigen's default).
///
/// The stopping criterion checks that the maximum coordinate update
/// @f$ \max_j |w_j^{\mathrm{new}} - w_j^{\mathrm{old}}| @f$
/// is smaller than `tol`.
///
/// @note **scikit-learn parity gaps:** The following sklearn constructor
///   parameters are not yet supported: `precompute`, `copy_X`, `warm_start`,
///   `positive`, `random_state`, `selection`.
///   The following sklearn fitted attributes are not yet exposed:
///   `n_iter_`, `dual_gap_`, `sparse_coef_`, `n_features_in_`,
///   `feature_names_in_`.
///
/// ### Examples
///
/// @snippet elastic_net.cpp example_elasticnet
template <typename Scalar = double>
class ElasticNet
    : public Predictor<ElasticNet<Scalar>, Scalar> {
public:
    using Base = Predictor<ElasticNet<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;

    // Make the dense base-class fit overload visible alongside the
    // sparse fit overload added below.
    using Base::fit;

    /// @brief Construct an ElasticNet estimator.
    ///
    /// @param alpha Constant that multiplies the penalty terms (`Scalar`, default `1`).
    ///   `alpha = 0` is equivalent to an ordinary least square, solved by
    ///   LinearRegression. For numerical reasons, using `alpha = 0` with
    ///   ElasticNet is not advised.
    /// @param l1_ratio The ElasticNet mixing parameter (`Scalar`, default `0.5`).
    ///   With `0 <= l1_ratio <= 1`:
    ///   `l1_ratio = 0` → pure L2 penalty (Ridge);
    ///   `l1_ratio = 1` → pure L1 penalty (Lasso);
    ///   `0 < l1_ratio < 1` → combination of L1 and L2.
    /// @param fit_intercept Whether the intercept should be estimated (`bool`, default `true`).
    ///   If `false`, the data is assumed to be already centered.
    /// @param max_iter The maximum number of iterations (`int`, default `1000`).
    /// @param tol The tolerance for the optimization (`Scalar`, default `1e-4`):
    ///   if the maximum coordinate update is smaller than `tol`, the solver stops.
    explicit ElasticNet(Scalar alpha = Scalar{1}, Scalar l1_ratio = Scalar{0.5},
                        bool fit_intercept = true,
                        int max_iter = 1000, Scalar tol = Scalar{1e-4})
        : alpha_(alpha), l1_ratio_(l1_ratio), fit_intercept_(fit_intercept),
          max_iter_(max_iter), tol_(tol) {}

    /// @brief Parameter vector @f$w@f$ (1 × n_features).
    ///
    /// Estimated coefficients for the linear regression problem.
    /// If `fit_intercept` is `true`, the coefficients correspond to the
    /// centered data.
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

    /// @brief Fit model with coordinate descent.
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
    ElasticNet& fit_impl(const Eigen::Ref<const MatrixType>& X,
                         const Eigen::Ref<const VectorType>& y) {
        internal::check_non_empty(X);
        this->n_features_in_ = X.cols();
        const Eigen::Index n = X.rows();
        const Eigen::Index p = X.cols();

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

        VectorType col_norms_sq(p);
        for (Eigen::Index j = 0; j < p; ++j) {
            col_norms_sq(j) = X_c.col(j).squaredNorm();
        }

        Scalar l1_penalty = static_cast<Scalar>(n) * alpha_ * l1_ratio_;
        Scalar l2_penalty = static_cast<Scalar>(n) * alpha_ * (Scalar{1} - l1_ratio_);

        coef_ = RowVectorType::Zero(p);
        VectorType residual = y_c;

        for (int iter = 0; iter < max_iter_; ++iter) {
            Scalar max_change{0};

            for (Eigen::Index j = 0; j < p; ++j) {
                Scalar denom = col_norms_sq(j) + l2_penalty;
                if (denom < std::numeric_limits<Scalar>::epsilon()) continue;

                Scalar old_w = coef_(j);
                Scalar rho = X_c.col(j).dot(residual) + col_norms_sq(j) * old_w;

                coef_(j) = soft_threshold(rho, l1_penalty) / denom;

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

    // -- Sparse-aware overload (v1.1.0 §3.2) --------------------------------

    /// @brief Fit ElasticNet on a sparse design matrix without densifying X.
    ///
    /// Implements coordinate descent with implicit centring (same trick
    /// as `Lasso::fit(SparseMatrix, ...)`). The CD denominator is
    /// extended by the L2 penalty
    /// @f$ \mathrm{denom}_j = \|X_c[:, j]\|^2 + n\,\alpha\,(1 - \rho) @f$
    /// where @f$\rho@f$ is `l1_ratio_`. The L1 soft-thresholding numerator
    /// uses @f$ n\,\alpha\,\rho @f$, matching sklearn's parameterisation.
    /// We maintain `r_raw = y_c - X*w` and a scalar
    /// `shift = Σ_k w_k · x̄_k` to recover the centred dot product
    /// @f$ X_c[:, j] \cdot r^{centered}
    ///   = X[:, j] \cdot r^{raw} + n\,\bar{x}_j\,\mathrm{shift} @f$.
    ///
    /// `sample_weight`, `precompute`, `positive`, `random_state`,
    /// `selection`, and `warm_start` are documented parity gaps.
    template <int Options, typename StorageIndex>
    ElasticNet& fit(
        const Eigen::SparseMatrix<Scalar, Options, StorageIndex>& X,
        const Eigen::Ref<const VectorType>& y) {
        if (X.rows() == 0 || X.cols() == 0) {
            throw std::invalid_argument("ElasticNet.fit: empty sparse matrix.");
        }
        if (X.rows() != y.size()) {
            throw std::invalid_argument(
                "ElasticNet.fit: X has " + std::to_string(X.rows()) +
                " rows but y has " + std::to_string(y.size()) + ".");
        }
        this->n_features_in_ = X.cols();

        using ColSparse =
            Eigen::SparseMatrix<Scalar, Eigen::ColMajor, StorageIndex>;
        const ColSparse Xc = X;
        const Eigen::Index n = Xc.rows();
        const Eigen::Index p = Xc.cols();

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

        VectorType col_norms_sq(p);
        for (Eigen::Index j = 0; j < p; ++j) {
            col_norms_sq(j) = fit_intercept_
                ? raw_norm_sq(j)
                  - static_cast<Scalar>(n) * x_mean(j) * x_mean(j)
                : raw_norm_sq(j);
            if (col_norms_sq(j) < Scalar{0}) col_norms_sq(j) = Scalar{0};
        }

        const Scalar l1_penalty =
            static_cast<Scalar>(n) * alpha_ * l1_ratio_;
        const Scalar l2_penalty =
            static_cast<Scalar>(n) * alpha_ * (Scalar{1} - l1_ratio_);

        coef_ = RowVectorType::Zero(p);
        VectorType r_raw = fit_intercept_
            ? VectorType((y.array() - y_mean).matrix())
            : VectorType(y);
        Scalar shift{0};

        for (int iter = 0; iter < max_iter_; ++iter) {
            Scalar max_change{0};

            for (Eigen::Index j = 0; j < p; ++j) {
                const Scalar denom = col_norms_sq(j) + l2_penalty;
                if (denom < std::numeric_limits<Scalar>::epsilon()) continue;

                const Scalar old_w = coef_(j);
                Scalar dot{0};
                for (typename ColSparse::InnerIterator it(Xc, j); it; ++it) {
                    dot += it.value() * r_raw(it.row());
                }
                Scalar rho = dot + col_norms_sq(j) * old_w;
                if (fit_intercept_) {
                    rho += static_cast<Scalar>(n) * x_mean(j) * shift;
                }
                const Scalar new_w = soft_threshold(rho, l1_penalty) / denom;
                const Scalar delta = new_w - old_w;

                if (delta != Scalar{0}) {
                    for (typename ColSparse::InnerIterator it(Xc, j);
                         it; ++it) {
                        r_raw(it.row()) -= delta * it.value();
                    }
                    if (fit_intercept_) shift += delta * x_mean(j);
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
    /// The coefficient of determination is defined as
    /// @f$ R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2} @f$.
    /// Best possible score is 1.0; it can be negative if the model is
    /// arbitrarily worse than predicting the mean.
    ///
    /// @param X Test samples of shape (n_samples, n_features).
    /// @param y True values of shape (n_samples,).
    /// @return @f$R^2@f$ score.
    /// @throws std::runtime_error if the model has not been fitted.
    ///
    /// @note **sklearn parity gap:** `sample_weight` parameter is
    ///   not yet supported.
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
    Scalar l1_ratio_;
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

#endif // SKIGEN_LINEAR_MODEL_ELASTIC_NET_H
