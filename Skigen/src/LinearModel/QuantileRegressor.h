// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

#ifndef SKIGEN_LINEAR_MODEL_QUANTILE_REGRESSOR_H
#define SKIGEN_LINEAR_MODEL_QUANTILE_REGRESSOR_H

#include "../Core/Base.h"
#include "../Core/Validation.h"
#include "Detail/LinearProgram.h"

#include <Eigen/Core>
#include <stdexcept>
#include <string>

namespace Skigen {

/// @defgroup Algo_QuantileRegressor QuantileRegressor
/// @ingroup LinearModels
/// @brief Linear quantile regression via linear programming.
/// @{

/// @brief Linear regression model that predicts conditional quantiles.
///
/// Minimises the pinball (quantile) loss with an L1 penalty:
/// @f[
///   \min_{w, b}\; \frac{1}{n}\sum_i \rho_\tau\!\big(y_i - x_i^\top w - b\big)
///     + \alpha \lVert w \rVert_1,
/// @f]
/// where @f$\rho_\tau(r) = \max(\tau r, (\tau - 1) r)@f$ is the pinball
/// loss for quantile @f$\tau@f$. The problem is reformulated as a linear
/// program and solved with a native primal-dual interior-point method, so
/// no external LP backend is required.
///
/// Mirrors
/// [sklearn.linear_model.QuantileRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.QuantileRegressor.html).
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `quantile` | `Scalar` | `0.5` | The quantile to predict, in `(0, 1)`. |
/// | `alpha` | `Scalar` | `1.0` | L1 regularisation strength. `0` disables regularisation. |
/// | `fit_intercept` | `bool` | `true` | Whether to fit an unpenalised intercept. |
/// | `solver` | `std::string` | `"interior-point"` | LP backend (only the native interior-point solver is available). |
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type | Description |
/// |----------|------|-------------|
/// | `coef()` | `RowVectorType` | Coefficient vector @f$w@f$ (1 × n_features). |
/// | `intercept()` | `Scalar` | Bias term @f$b@f$ (0 when `fit_intercept = false`). |
/// | `n_iter()` | `int` | Interior-point iterations performed. |
///
/// ### See also
///
/// - Skigen::Lasso — L1-penalised least-squares regression.
/// - Skigen::HuberRegressor analogue (robust regression).
///
/// ### Limitations relative to scikit-learn
///
/// Only the native interior-point `solver` is provided; sklearn's
/// `"highs"`, `"highs-ds"`, `"highs-ipm"`, and `"revised simplex"` solver
/// names are not selectable (passing them raises). `solver_options` and
/// `sample_weight` are not honoured. Sparse input falls back to a dense
/// solve.
///
/// ### Examples
///
/// @snippet quantile_regressor.cpp example_quantile_regressor
template <typename Scalar = double>
class QuantileRegressor
    : public Predictor<QuantileRegressor<Scalar>, Scalar> {
public:
    using Base = Predictor<QuantileRegressor<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;
    using Base::fit;

    /// @brief Construct a QuantileRegressor.
    ///
    /// @param quantile Target quantile in `(0, 1)` (`Scalar`, default `0.5`).
    /// @param alpha L1 regularisation strength (`Scalar`, default `1.0`).
    /// @param fit_intercept Whether to fit an intercept (`bool`, default `true`).
    /// @param solver LP backend name (`std::string`, default `"interior-point"`).
    explicit QuantileRegressor(Scalar quantile = Scalar{0.5},
                               Scalar alpha = Scalar{1},
                               bool fit_intercept = true,
                               std::string solver = "interior-point")
        : quantile_(quantile),
          alpha_(alpha),
          fit_intercept_(fit_intercept),
          solver_(std::move(solver)) {}

    // -- Accessors ----------------------------------------------------------

    [[nodiscard]] Scalar quantile() const noexcept { return quantile_; }
    [[nodiscard]] Scalar alpha() const noexcept { return alpha_; }
    [[nodiscard]] bool fit_intercept() const noexcept { return fit_intercept_; }
    [[nodiscard]] const std::string& solver() const noexcept {
        return solver_;
    }

    [[nodiscard]] const RowVectorType& coef() const {
        this->check_is_fitted(); return coef_;
    }
    [[nodiscard]] Scalar intercept() const {
        this->check_is_fitted(); return intercept_;
    }
    [[nodiscard]] int n_iter() const {
        this->check_is_fitted(); return n_iter_;
    }

    SKIGEN_PARAMS(
        (quantile,      quantile_,      double),
        (alpha,         alpha_,         double),
        (fit_intercept, fit_intercept_, bool),
        (solver,        solver_,        std::string))

    // -- Fit / Predict ------------------------------------------------------

    /// @brief Fit the quantile regression LP.
    ///
    /// Builds the standard-form LP
    /// @f$\min_{z \ge 0} c^\top z@f$ s.t. @f$A z = y@f$ with
    /// @f$z = (w^+, w^-, b^+, b^-, u, v)@f$ and recovers
    /// @f$w = w^+ - w^-@f$, @f$b = b^+ - b^-@f$.
    QuantileRegressor& fit_impl(const Eigen::Ref<const MatrixType>& X,
                                const Eigen::Ref<const VectorType>& y) {
        internal::check_non_empty(X);
        internal::check_consistent_length(X, y);
        if (quantile_ <= Scalar{0} || quantile_ >= Scalar{1}) {
            throw std::invalid_argument(
                "QuantileRegressor: quantile must be in (0, 1); got " +
                std::to_string(quantile_) + ".");
        }
        if (solver_ != "interior-point") {
            throw std::invalid_argument(
                "QuantileRegressor: only solver='interior-point' is "
                "supported; got '" + solver_ + "'.");
        }
        this->n_features_in_ = X.cols();

        const Eigen::Index n = X.rows();
        const Eigen::Index p = X.cols();
        const Eigen::Index n_int = fit_intercept_ ? 1 : 0;

        // Variable layout: [ w+ (p) | w- (p) | b+ (n_int) | b- (n_int)
        //                    | u (n) | v (n) ].
        const Eigen::Index n_w = 2 * p;
        const Eigen::Index n_b = 2 * n_int;
        const Eigen::Index n_vars = n_w + n_b + 2 * n;

        // Equality constraints: X w + b + u - v = y, i.e.
        //   [ X  -X  1  -1  I  -I ] z = y.
        MatrixType A = MatrixType::Zero(n, n_vars);
        A.block(0, 0, n, p) = X;
        A.block(0, p, n, p) = -X;
        Eigen::Index off = n_w;
        if (fit_intercept_) {
            A.block(0, off, n, 1) = VectorType::Ones(n);
            A.block(0, off + 1, n, 1) = -VectorType::Ones(n);
            off += 2;
        }
        A.block(0, off, n, n) = MatrixType::Identity(n, n);
        A.block(0, off + n, n, n) = -MatrixType::Identity(n, n);

        // Cost: alpha on |w| (= w+ + w-), pinball weights on (u, v).
        // sklearn scales the data-fit term by 1/n_samples.
        VectorType c = VectorType::Zero(n_vars);
        const Scalar an = alpha_ * static_cast<Scalar>(n);
        for (Eigen::Index j = 0; j < n_w; ++j) c(j) = an;
        // intercept is unpenalised (c stays 0 on b+ / b-).
        const Eigen::Index u_off = n_w + n_b;
        for (Eigen::Index i = 0; i < n; ++i) {
            c(u_off + i) = quantile_;                 // u: positive residual
            c(u_off + n + i) = Scalar{1} - quantile_;  // v: negative residual
        }

        const auto sol = internal::solve_standard_lp<Scalar>(A, y, c);
        n_iter_ = sol.n_iter;

        coef_ = RowVectorType::Zero(p);
        for (Eigen::Index j = 0; j < p; ++j)
            coef_(j) = sol.x(j) - sol.x(p + j);
        intercept_ = fit_intercept_
            ? sol.x(n_w) - sol.x(n_w + 1)
            : Scalar{0};

        this->fitted_ = true;
        return *this;
    }

    [[nodiscard]] VectorType predict_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        return (X * coef_.transpose()).array() + intercept_;
    }

    /// @brief R² coefficient of determination on test data.
    [[nodiscard]] Scalar score_impl(
        const Eigen::Ref<const MatrixType>& X,
        const Eigen::Ref<const VectorType>& y) const {
        const VectorType yhat = predict_impl(X);
        const Scalar ss_res = (y - yhat).squaredNorm();
        const Scalar ss_tot = (y.array() - y.mean()).matrix().squaredNorm();
        if (ss_tot == Scalar{0}) return Scalar{0};
        return Scalar{1} - ss_res / ss_tot;
    }

private:
    Scalar quantile_;
    Scalar alpha_;
    bool fit_intercept_;
    std::string solver_;

    RowVectorType coef_;
    Scalar intercept_{0};
    int n_iter_{0};
};

/// @}

}  // namespace Skigen

#endif  // SKIGEN_LINEAR_MODEL_QUANTILE_REGRESSOR_H
