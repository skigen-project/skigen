// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_LINEAR_MODEL_BAYESIAN_RIDGE_H
#define SKIGEN_LINEAR_MODEL_BAYESIAN_RIDGE_H

#include "../Core/Base.h"
#include "../Core/Validation.h"

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/SparseCore>

#include <cmath>
#include <iostream>
#include <limits>
#include <numbers>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace Skigen {

/// @cond INTERNAL
struct return_std_t {};
/// @endcond
/// @brief Convenience tag value: `est.predict(X, with_std)`.
inline constexpr return_std_t with_std{};

/// @defgroup Algo_BayesianRidge Bayesian Ridge Regression
/// @ingroup LinearModels
/// @brief Bayesian linear regression with Gamma priors over precision parameters.
/// @{

/// @brief Bayesian ridge regression.
///
/// Fit a Bayesian ridge model with Gamma hyper-priors on the noise
/// precision @f$\alpha@f$ and the weight precision @f$\lambda@f$. The
/// implementation follows Appendix A of (Tipping, 2001) with MacKay's
/// (1992) update rules for the regularization parameters.
///
/// Mirrors
/// [sklearn.linear_model.BayesianRidge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html).
///
/// The objective maximised by the iterative procedure is the (log) marginal
/// likelihood
///
/// @f[
///   \log p(y \mid \alpha, \lambda)
///     = \tfrac{1}{2}\bigl[ N\log\alpha + p\log\lambda
///       - \alpha \lVert y - Xw \rVert^2
///       - \lambda \lVert w \rVert^2
///       + \log |\Sigma|
///       - N \log 2\pi \bigr] + \mathrm{const.}
/// @f]
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `max_iter` | `int` | `300` | Maximum number of evidence-maximisation iterations. |
/// | `tol` | `Scalar` | `1e-3` | Stop when @f$\sum |w_{\text{new}} - w_{\text{old}}| < \text{tol}@f$. |
/// | `alpha_1` | `Scalar` | `1e-6` | Shape parameter of the Gamma prior over @f$\alpha@f$. |
/// | `alpha_2` | `Scalar` | `1e-6` | Inverse-scale (rate) of the Gamma prior over @f$\alpha@f$. |
/// | `lambda_1` | `Scalar` | `1e-6` | Shape parameter of the Gamma prior over @f$\lambda@f$. |
/// | `lambda_2` | `Scalar` | `1e-6` | Inverse-scale (rate) of the Gamma prior over @f$\lambda@f$. |
/// | `alpha_init` | `optional<Scalar>` | `nullopt` | Initial value for @f$\alpha@f$ (default `1/Var(y)`). |
/// | `lambda_init` | `optional<Scalar>` | `nullopt` | Initial value for @f$\lambda@f$ (default `1`). |
/// | `compute_score` | `bool` | `false` | If true, log-marginal-likelihood is recorded each iteration. |
/// | `fit_intercept` | `bool` | `true` | Whether the intercept should be estimated. |
/// | `copy_X` | `bool` | `true` | **Deprecated** no-op (X is never overwritten). |
/// | `verbose` | `bool` | `false` | Print convergence information. |
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type | Description |
/// |----------|------|-------------|
/// | `coef()` | `RowVectorType` | Posterior mean of the weights, shape (1 × n_features). |
/// | `intercept()` | `Scalar` | Independent term in the decision function. |
/// | `alpha()` | `Scalar` | Estimated precision of the noise. |
/// | `lambda_()` | `Scalar` | Estimated precision of the weights. |
/// | `sigma()` | `MatrixType` | Posterior covariance, shape (n_features × n_features). |
/// | `scores()` | `VectorType` | Log-marginal-likelihood per iteration (only if `compute_score=true`). |
/// | `n_iter()` | `int` | Number of iterations executed. |
/// | `X_offset()` | `RowVectorType` | Centering offsets for X (zeros when `fit_intercept=false`). |
/// | `X_scale()` | `RowVectorType` | Per-feature scaling (always ones in this implementation). |
/// | `y_offset()` | `Scalar` | Centering offset for y. |
///
/// ### See also
///
/// - Skigen::ARDRegression — Bayesian ARD regression with per-feature precisions.
/// - Skigen::Ridge — Frequentist L2 regression.
///
/// ### References
///
/// - M. E. Tipping, *Sparse Bayesian Learning and the Relevance Vector Machine*, JMLR 2001.
/// - D. J. C. MacKay, *Bayesian Interpolation*, Neural Computation, 1992.
///
/// ### Limitations relative to scikit-learn `sample_weight` in `fit()` is not
///   supported. `copy_X` is accepted but ignored (no-op). Sparse inputs are
///   not supported. `n_features_in_` is exposed via `n_features_in()` but
///   `feature_names_in_` is not.
///
/// ### Examples
///
/// @snippet bayesian_ridge.cpp example_bayesian_ridge
template <typename Scalar = double>
class BayesianRidge
    : public Predictor<BayesianRidge<Scalar>, Scalar> {
public:
    using Base = Predictor<BayesianRidge<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;
    using typename Base::ScalarType;
    using Base::fit;

    /// @brief Construct a Bayesian Ridge estimator with the given hyper-parameters.
    explicit BayesianRidge(int max_iter = 300,
                           Scalar tol = Scalar{1e-3},
                           Scalar alpha_1 = Scalar{1e-6},
                           Scalar alpha_2 = Scalar{1e-6},
                           Scalar lambda_1 = Scalar{1e-6},
                           Scalar lambda_2 = Scalar{1e-6},
                           std::optional<Scalar> alpha_init = std::nullopt,
                           std::optional<Scalar> lambda_init = std::nullopt,
                           bool compute_score = false,
                           bool fit_intercept = true,
                           bool copy_X = true,
                           bool verbose = false)
        : max_iter_(max_iter),
          tol_(tol),
          alpha_1_(alpha_1),
          alpha_2_(alpha_2),
          lambda_1_(lambda_1),
          lambda_2_(lambda_2),
          alpha_init_(alpha_init),
          lambda_init_(lambda_init),
          compute_score_(compute_score),
          fit_intercept_(fit_intercept),
          copy_X_(copy_X),
          verbose_(verbose) {
        (void)copy_X_;  // silence unused warning when no-op
    }

    // -- Hyper-parameter accessors ------------------------------------------

    [[nodiscard]] int max_iter() const noexcept { return max_iter_; }
    [[nodiscard]] Scalar tol() const noexcept { return tol_; }
    [[nodiscard]] Scalar alpha_1() const noexcept { return alpha_1_; }
    [[nodiscard]] Scalar alpha_2() const noexcept { return alpha_2_; }
    [[nodiscard]] Scalar lambda_1() const noexcept { return lambda_1_; }
    [[nodiscard]] Scalar lambda_2() const noexcept { return lambda_2_; }
    [[nodiscard]] std::optional<Scalar> alpha_init() const noexcept { return alpha_init_; }
    [[nodiscard]] std::optional<Scalar> lambda_init() const noexcept { return lambda_init_; }
    [[nodiscard]] bool compute_score() const noexcept { return compute_score_; }
    [[nodiscard]] bool fit_intercept() const noexcept { return fit_intercept_; }
    [[nodiscard]] bool verbose() const noexcept { return verbose_; }

    // -- Fitted-attribute accessors -----------------------------------------

    /// @brief Posterior mean of the weights, shape (1 × n_features).
    [[nodiscard]] const RowVectorType& coef() const {
        this->check_is_fitted(); return coef_;
    }
    /// @brief Independent term in the decision function.
    [[nodiscard]] Scalar intercept() const {
        this->check_is_fitted(); return intercept_;
    }
    /// @brief Estimated noise precision @f$\alpha@f$.
    [[nodiscard]] Scalar alpha() const {
        this->check_is_fitted(); return alpha_;
    }
    /// @brief Estimated weight precision @f$\lambda@f$.
    ///
    /// Method name uses a trailing underscore to match sklearn's
    /// `lambda_` attribute (the keyword `lambda` is reserved in C++).
    [[nodiscard]] Scalar lambda_() const {
        this->check_is_fitted(); return lambda_val_;
    }
    /// @brief Posterior covariance of the weights, shape (n_features × n_features).
    [[nodiscard]] const MatrixType& sigma() const {
        this->check_is_fitted(); return sigma_;
    }
    /// @brief Log-marginal-likelihood values recorded during fitting.
    [[nodiscard]] const VectorType& scores() const {
        this->check_is_fitted(); return scores_;
    }
    /// @brief Number of iterations executed.
    [[nodiscard]] int n_iter() const {
        this->check_is_fitted(); return n_iter_;
    }
    /// @brief Centering offsets used for X (zeros if `fit_intercept=false`).
    [[nodiscard]] const RowVectorType& X_offset() const {
        this->check_is_fitted(); return X_offset_;
    }
    /// @brief Per-feature scaling (always ones in this implementation).
    [[nodiscard]] const RowVectorType& X_scale() const {
        this->check_is_fitted(); return X_scale_;
    }
    /// @brief Centering offset used for y.
    [[nodiscard]] Scalar y_offset() const {
        this->check_is_fitted(); return y_offset_;
    }

    SKIGEN_PARAMS(
        (max_iter,      max_iter_,      int),
        (tol,           tol_,           double),
        (alpha_1,       alpha_1_,       double),
        (alpha_2,       alpha_2_,       double),
        (lambda_1,      lambda_1_,      double),
        (lambda_2,      lambda_2_,      double),
        (compute_score, compute_score_, bool),
        (fit_intercept, fit_intercept_, bool))

    // -- Implementation (called by CRTP base) --------------------------------

    /// @brief Fit the Bayesian Ridge model via SVD-based evidence maximisation.
    BayesianRidge& fit_impl(const Eigen::Ref<const MatrixType>& X,
                            const Eigen::Ref<const VectorType>& y) {
        internal::check_non_empty(X);
        internal::check_consistent_length(X, y);

        const Eigen::Index n_samples  = X.rows();
        const Eigen::Index n_features = X.cols();
        this->n_features_in_ = n_features;

        // ---- Centering -----------------------------------------------------
        MatrixType Xc;
        VectorType yc;
        X_scale_ = RowVectorType::Ones(n_features);
        if (fit_intercept_) {
            X_offset_ = X.colwise().mean();
            y_offset_ = y.mean();
            Xc = X.rowwise() - X_offset_;
            yc = y.array() - y_offset_;
        } else {
            X_offset_ = RowVectorType::Zero(n_features);
            y_offset_ = Scalar{0};
            Xc = X;
            yc = y;
        }

        // ---- SVD of centered X --------------------------------------------
        // We need V (full, n×p) for the posterior covariance even when
        // n_samples >= n_features, and U (thin) for the small-sample branch.
        const bool full_v = (n_samples < n_features);
        Eigen::JacobiSVD<MatrixType> svd(
            Xc,
            Eigen::ComputeThinU |
                (full_v ? Eigen::ComputeFullV : Eigen::ComputeThinV));

        const VectorType S = svd.singularValues();
        const Eigen::Index K = S.size();
        VectorType eigen_vals = S.array().square();              // size K
        VectorType eigen_vals_full = VectorType::Zero(n_features);
        eigen_vals_full.head(K) = eigen_vals;
        const MatrixType& U = svd.matrixU();   // (n_samples × K)
        const MatrixType& V_full = svd.matrixV();  // (n_features × n_features) or thin
        // Vh : (K × n_features) — top K rows of V^T
        MatrixType Vh = V_full.leftCols(K).transpose();

        // ---- Initialisation ------------------------------------------------
        constexpr Scalar eps = std::numeric_limits<Scalar>::epsilon();
        const Scalar y_var = variance(y);
        Scalar alpha_v = alpha_init_.has_value() ? *alpha_init_
                                                 : Scalar{1} / (y_var + eps);
        Scalar lambda_v = lambda_init_.has_value() ? *lambda_init_ : Scalar{1};

        const VectorType XT_y = Xc.transpose() * yc;

        std::vector<Scalar> score_log;
        if (compute_score_) score_log.reserve(static_cast<std::size_t>(max_iter_) + 1);

        VectorType coef = VectorType::Zero(n_features);
        VectorType coef_old;
        Scalar sse = Scalar{0};
        const auto sw_sum = static_cast<Scalar>(n_samples);

        int iter_done = 0;
        for (int iter = 0; iter < max_iter_; ++iter) {
            update_coef(Xc, yc, XT_y, U, Vh, eigen_vals,
                        alpha_v, lambda_v, coef, sse);

            if (compute_score_) {
                score_log.push_back(log_marginal_likelihood(
                    n_samples, n_features, sw_sum, eigen_vals,
                    alpha_v, lambda_v, coef, sse));
            }

            // MacKay (1992) updates
            Scalar gamma = (alpha_v * eigen_vals.array() /
                            (lambda_v + alpha_v * eigen_vals.array())).sum();
            lambda_v = (gamma + Scalar{2} * lambda_1_) /
                       (coef.squaredNorm() + Scalar{2} * lambda_2_);
            alpha_v = (sw_sum - gamma + Scalar{2} * alpha_1_) /
                      (sse + Scalar{2} * alpha_2_);

            iter_done = iter;
            if (iter != 0 && (coef_old - coef).cwiseAbs().sum() < tol_) {
                if (verbose_) {
                    std::cout << "Convergence after " << iter << " iterations\n";
                }
                break;
            }
            coef_old = coef;
        }

        n_iter_ = iter_done + 1;
        alpha_ = alpha_v;
        lambda_val_ = lambda_v;

        // Final coefficient pass with converged alpha/lambda
        update_coef(Xc, yc, XT_y, U, Vh, eigen_vals,
                    alpha_v, lambda_v, coef, sse);
        if (compute_score_) {
            score_log.push_back(log_marginal_likelihood(
                n_samples, n_features, sw_sum, eigen_vals,
                alpha_v, lambda_v, coef, sse));
        }

        // Posterior covariance:
        // sigma_ = V_full * diag(1 / (alpha * eig_full + lambda)) * V_full^T
        VectorType inv_diag =
            (alpha_v * eigen_vals_full.array() + lambda_v).inverse();
        sigma_ = V_full *
                 inv_diag.asDiagonal() *
                 V_full.transpose();

        coef_ = coef.transpose();

        if (compute_score_) {
            scores_ = VectorType(static_cast<Eigen::Index>(score_log.size()));
            for (std::size_t i = 0; i < score_log.size(); ++i) {
                scores_(static_cast<Eigen::Index>(i)) = score_log[i];
            }
        } else {
            scores_.resize(0);
        }

        if (fit_intercept_) {
            intercept_ = y_offset_ - X_offset_.dot(coef);
        } else {
            intercept_ = Scalar{0};
        }

        this->fitted_ = true;
        return *this;
    }

    /// @brief Fit from a sparse design matrix (densifies internally).
    template <int Options, typename StorageIndex>
    BayesianRidge& fit(const Eigen::SparseMatrix<Scalar, Options, StorageIndex>& X,
                       const Eigen::Ref<const VectorType>& y) {
        if (X.rows() == 0 || X.cols() == 0)
            throw std::invalid_argument("BayesianRidge.fit: empty sparse matrix.");
        MatrixType Xd = MatrixType(X);
        return fit_impl(Xd, y);
    }

    /// @brief Predictive mean @f$\hat{y} = X w + b@f$.
    [[nodiscard]] VectorType predict_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        return (X * coef_.transpose()).array() + intercept_;
    }

    /// @brief @f$R^2@f$ score.
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

    /// @brief Predictive mean and standard deviation.
    ///
    /// Tag-dispatch overload selected by passing `Skigen::with_std`. Returns
    /// `{y_mean, y_std}`. The standard deviation is the marginal predictive
    /// std-dev:
    /// @f$ \sigma_{\hat{y}}^2(x_*) = x_*^\top \Sigma\, x_* + 1/\alpha @f$.
    [[nodiscard]] std::pair<VectorType, VectorType> predict(
        const Eigen::Ref<const MatrixType>& X, std::true_type) const {
        this->check_is_fitted();
        this->validate_feature_count(X);
        VectorType y_mean = predict_impl(X);
        // sigmas_squared_data[i] = X[i] . sigma . X[i]^T
        MatrixType XS = X * sigma_;
        VectorType sigmas_sq =
            (XS.array() * X.array()).rowwise().sum();
        VectorType y_std = (sigmas_sq.array() + Scalar{1} / alpha_).sqrt();
        return {std::move(y_mean), std::move(y_std)};
    }

    /// @brief Tag overload (`with_std`) — calls `predict(X, std::true_type{})`.
    [[nodiscard]] std::pair<VectorType, VectorType> predict(
        const Eigen::Ref<const MatrixType>& X, return_std_t) const {
        return predict(X, std::true_type{});
    }

    // Bring base-class `predict(X)` into scope so unqualified lookups still work.
    using Base::predict;

private:
    static Scalar variance(const Eigen::Ref<const VectorType>& y) {
        if (y.size() == 0) return Scalar{0};
        const Scalar mean = y.mean();
        return (y.array() - mean).square().sum() / static_cast<Scalar>(y.size());
    }

    static void update_coef(const MatrixType& X,
                            const VectorType& y,
                            const VectorType& XT_y,
                            const MatrixType& U,
                            const MatrixType& Vh,
                            const VectorType& eigen_vals,
                            Scalar alpha_v,
                            Scalar lambda_v,
                            VectorType& coef_out,
                            Scalar& sse_out) {
        const Eigen::Index n_samples  = X.rows();
        const Eigen::Index n_features = X.cols();
        VectorType denom = eigen_vals.array() + lambda_v / alpha_v;
        if (n_samples > n_features) {
            // coef = Vh^T * diag(1/denom) * Vh * XT_y
            VectorType tmp = Vh * XT_y;                 // K
            tmp.array() /= denom.array();
            coef_out = Vh.transpose() * tmp;            // p
        } else {
            // coef = X^T * U * diag(1/denom) * U^T * y
            VectorType tmp = U.transpose() * y;         // K
            tmp.array() /= denom.array();
            coef_out = X.transpose() * (U * tmp);       // p
        }
        sse_out = (y - X * coef_out).squaredNorm();
    }

    Scalar log_marginal_likelihood(Eigen::Index n_samples,
                                   Eigen::Index n_features,
                                   Scalar sw_sum,
                                   const VectorType& eigen_vals,
                                   Scalar alpha_v,
                                   Scalar lambda_v,
                                   const VectorType& coef,
                                   Scalar sse) const {
        Scalar logdet_sigma;
        if (n_samples > n_features) {
            logdet_sigma = -((lambda_v + alpha_v * eigen_vals.array()).log()).sum();
        } else {
            // logdet over n_features eigenvalues, padded with lambda_v
            VectorType vals = VectorType::Constant(n_features, lambda_v);
            const Eigen::Index k =
                std::min<Eigen::Index>(n_samples, eigen_vals.size());
            vals.head(k).array() += alpha_v * eigen_vals.head(k).array();
            logdet_sigma = -(vals.array().log()).sum();
        }
        using std::log;
        Scalar score = lambda_1_ * log(lambda_v) - lambda_2_ * lambda_v;
        score += alpha_1_ * log(alpha_v) - alpha_2_ * alpha_v;
        score += Scalar{0.5} * (
            static_cast<Scalar>(n_features) * log(lambda_v)
            + sw_sum * log(alpha_v)
            - alpha_v * sse
            - lambda_v * coef.squaredNorm()
            + logdet_sigma
            - sw_sum * log(Scalar{2} * std::numbers::pi_v<Scalar>));
        return score;
    }

    // ---- Hyper-parameters -------------------------------------------------
    int max_iter_;
    Scalar tol_;
    Scalar alpha_1_;
    Scalar alpha_2_;
    Scalar lambda_1_;
    Scalar lambda_2_;
    std::optional<Scalar> alpha_init_;
    std::optional<Scalar> lambda_init_;
    bool compute_score_;
    bool fit_intercept_;
    bool copy_X_;
    bool verbose_;

    // ---- Fitted attributes ------------------------------------------------
    RowVectorType coef_;
    Scalar intercept_ = Scalar{0};
    Scalar alpha_ = Scalar{0};
    Scalar lambda_val_ = Scalar{0};
    MatrixType sigma_;
    VectorType scores_;
    int n_iter_ = 0;
    RowVectorType X_offset_;
    RowVectorType X_scale_;
    Scalar y_offset_ = Scalar{0};
};

/// @}

} // namespace Skigen

#endif // SKIGEN_LINEAR_MODEL_BAYESIAN_RIDGE_H
