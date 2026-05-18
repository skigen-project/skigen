// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_LINEAR_MODEL_ARD_REGRESSION_H
#define SKIGEN_LINEAR_MODEL_ARD_REGRESSION_H

#include "../Core/Base.h"
#include "../Core/Validation.h"
#include "BayesianRidge.h"  // for return_std_t / with_std

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include <cmath>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace Skigen {

/// @defgroup Algo_ARDRegression Bayesian ARD Regression
/// @ingroup LinearModels
/// @brief Automatic Relevance Determination — Bayesian linear regression with
///        a per-feature precision and feature pruning.
/// @{

/// @brief Bayesian Automatic Relevance Determination (ARD) regression.
///
/// Maintains a per-feature precision @f$\lambda_j@f$ and prunes features whose
/// precision exceeds `threshold_lambda` during the evidence-maximisation
/// iteration. Mirrors
/// [sklearn.linear_model.ARDRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html).
///
/// The posterior covariance over the surviving features is
///
/// @f[
///   \Sigma = \bigl(\operatorname{diag}(\lambda_{\text{keep}})
///                   + \alpha\, X_{\text{keep}}^\top X_{\text{keep}}\bigr)^{-1}
/// @f]
///
/// and the posterior mean is
/// @f$ w_{\text{keep}} = \alpha\, \Sigma\, X_{\text{keep}}^\top y @f$.
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `max_iter` | `int` | `300` | Maximum number of iterations. |
/// | `tol` | `Scalar` | `1e-3` | Stop when @f$\sum |w_{\text{new}} - w_{\text{old}}| < \text{tol}@f$. |
/// | `alpha_1` | `Scalar` | `1e-6` | Shape parameter of the Gamma prior over @f$\alpha@f$. |
/// | `alpha_2` | `Scalar` | `1e-6` | Inverse-scale (rate) of the Gamma prior over @f$\alpha@f$. |
/// | `lambda_1` | `Scalar` | `1e-6` | Shape parameter of the Gamma prior over @f$\lambda@f$. |
/// | `lambda_2` | `Scalar` | `1e-6` | Inverse-scale (rate) of the Gamma prior over @f$\lambda@f$. |
/// | `compute_score` | `bool` | `false` | If true, the objective value is recorded each iteration. |
/// | `threshold_lambda` | `Scalar` | `1e4` | Features with @f$\lambda_j > \text{threshold}@f$ are pruned. |
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
/// | `alpha()` | `Scalar` | Estimated noise precision. |
/// | `lambda_()` | `RowVectorType` | Per-feature weight precisions, shape (n_features,). |
/// | `sigma()` | `MatrixType` | Posterior covariance of the surviving features. |
/// | `scores()` | `VectorType` | Objective values per iteration (only if `compute_score=true`). |
/// | `n_iter()` | `int` | Number of iterations executed. |
/// | `X_offset()` / `X_scale()` / `y_offset()` | centering quantities |
///
/// ### See also
///
/// - Skigen::BayesianRidge — Bayesian regression with a single weight precision.
///
/// ### Limitations relative to scikit-learn `sample_weight` in `fit()` is not
///   supported. `copy_X` is accepted but ignored. Sparse inputs are not
///   supported. `feature_names_in_` is not exposed.
///
/// ### Examples
///
/// @snippet ard_regression.cpp example_ard_regression
template <typename Scalar = double>
class ARDRegression
    : public Predictor<ARDRegression<Scalar>, Scalar> {
public:
    using Base = Predictor<ARDRegression<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;
    using typename Base::ScalarType;
    using Base::fit;

    /// @brief Construct an ARD estimator with the given hyper-parameters.
    explicit ARDRegression(int max_iter = 300,
                           Scalar tol = Scalar{1e-3},
                           Scalar alpha_1 = Scalar{1e-6},
                           Scalar alpha_2 = Scalar{1e-6},
                           Scalar lambda_1 = Scalar{1e-6},
                           Scalar lambda_2 = Scalar{1e-6},
                           bool compute_score = false,
                           Scalar threshold_lambda = Scalar{1e4},
                           bool fit_intercept = true,
                           bool copy_X = true,
                           bool verbose = false)
        : max_iter_(max_iter),
          tol_(tol),
          alpha_1_(alpha_1),
          alpha_2_(alpha_2),
          lambda_1_(lambda_1),
          lambda_2_(lambda_2),
          compute_score_(compute_score),
          threshold_lambda_(threshold_lambda),
          fit_intercept_(fit_intercept),
          copy_X_(copy_X),
          verbose_(verbose) {
        (void)copy_X_;
    }

    // -- Hyper-parameter accessors ------------------------------------------

    [[nodiscard]] int max_iter() const noexcept { return max_iter_; }
    [[nodiscard]] Scalar tol() const noexcept { return tol_; }
    [[nodiscard]] Scalar alpha_1() const noexcept { return alpha_1_; }
    [[nodiscard]] Scalar alpha_2() const noexcept { return alpha_2_; }
    [[nodiscard]] Scalar lambda_1() const noexcept { return lambda_1_; }
    [[nodiscard]] Scalar lambda_2() const noexcept { return lambda_2_; }
    [[nodiscard]] bool compute_score() const noexcept { return compute_score_; }
    [[nodiscard]] Scalar threshold_lambda() const noexcept { return threshold_lambda_; }
    [[nodiscard]] bool fit_intercept() const noexcept { return fit_intercept_; }
    [[nodiscard]] bool verbose() const noexcept { return verbose_; }

    // -- Fitted-attribute accessors -----------------------------------------

    [[nodiscard]] const RowVectorType& coef() const {
        this->check_is_fitted(); return coef_;
    }
    [[nodiscard]] Scalar intercept() const {
        this->check_is_fitted(); return intercept_;
    }
    [[nodiscard]] Scalar alpha() const {
        this->check_is_fitted(); return alpha_;
    }
    /// @brief Per-feature weight precisions, shape (n_features,).
    [[nodiscard]] const RowVectorType& lambda_() const {
        this->check_is_fitted(); return lambda_vec_;
    }
    [[nodiscard]] const MatrixType& sigma() const {
        this->check_is_fitted(); return sigma_;
    }
    [[nodiscard]] const VectorType& scores() const {
        this->check_is_fitted(); return scores_;
    }
    [[nodiscard]] int n_iter() const {
        this->check_is_fitted(); return n_iter_;
    }
    [[nodiscard]] const RowVectorType& X_offset() const {
        this->check_is_fitted(); return X_offset_;
    }
    [[nodiscard]] const RowVectorType& X_scale() const {
        this->check_is_fitted(); return X_scale_;
    }
    [[nodiscard]] Scalar y_offset() const {
        this->check_is_fitted(); return y_offset_;
    }

    SKIGEN_PARAMS(
        (max_iter,          max_iter_,          int),
        (tol,               tol_,               double),
        (alpha_1,           alpha_1_,           double),
        (alpha_2,           alpha_2_,           double),
        (lambda_1,          lambda_1_,          double),
        (lambda_2,          lambda_2_,          double),
        (compute_score,     compute_score_,     bool),
        (threshold_lambda,  threshold_lambda_,  double),
        (fit_intercept,     fit_intercept_,     bool))

    // -- Implementation -----------------------------------------------------

    /// @brief Fit the ARD model via evidence maximisation with feature pruning.
    ARDRegression& fit_impl(const Eigen::Ref<const MatrixType>& X,
                            const Eigen::Ref<const VectorType>& y) {
        internal::check_non_empty(X);
        internal::check_consistent_length(X, y);
        if (X.rows() < 2) {
            throw std::invalid_argument(
                "ARDRegression requires at least 2 samples.");
        }

        const Eigen::Index n_samples  = X.rows();
        const Eigen::Index n_features = X.cols();
        this->n_features_in_ = n_features;

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

        VectorType coef = VectorType::Zero(n_features);
        VectorType lambda = VectorType::Ones(n_features);
        Eigen::Array<bool, Eigen::Dynamic, 1> keep_lambda =
            Eigen::Array<bool, Eigen::Dynamic, 1>::Constant(n_features, true);

        constexpr Scalar eps = std::numeric_limits<Scalar>::epsilon();
        const Scalar y_var = variance(y);
        Scalar alpha_v = Scalar{1} / (y_var + eps);

        std::vector<Scalar> score_log;
        if (compute_score_) score_log.reserve(static_cast<std::size_t>(max_iter_));

        VectorType coef_old = VectorType::Zero(n_features);
        MatrixType sigma;
        Scalar sse = Scalar{0};
        int iter_done = 0;

        for (int iter = 0; iter < max_iter_; ++iter) {
            // Build keep-indices
            std::vector<Eigen::Index> idx;
            idx.reserve(static_cast<std::size_t>(n_features));
            for (Eigen::Index j = 0; j < n_features; ++j) {
                if (keep_lambda(j)) idx.push_back(j);
            }
            const Eigen::Index k = static_cast<Eigen::Index>(idx.size());
            if (k == 0) break;

            MatrixType X_keep(n_samples, k);
            VectorType lambda_keep(k);
            for (Eigen::Index j = 0; j < k; ++j) {
                X_keep.col(j) = Xc.col(idx[static_cast<std::size_t>(j)]);
                lambda_keep(j) = lambda(idx[static_cast<std::size_t>(j)]);
            }

            // sigma = update_sigma
            sigma = update_sigma(X_keep, alpha_v, lambda_keep);

            // coef[keep] = alpha * sigma * X_keep^T * y
            VectorType coef_keep = alpha_v * (sigma * (X_keep.transpose() * yc));

            // Reset coef and copy back
            coef.setZero();
            for (Eigen::Index j = 0; j < k; ++j) {
                coef(idx[static_cast<std::size_t>(j)]) = coef_keep(j);
            }

            // sse and gamma
            VectorType resid = yc - Xc * coef;
            sse = resid.squaredNorm();
            VectorType sigma_diag = sigma.diagonal();
            VectorType gamma = VectorType::Ones(k).array()
                               - lambda_keep.array() * sigma_diag.array();
            // lambda update on keep
            for (Eigen::Index j = 0; j < k; ++j) {
                Eigen::Index gj = idx[static_cast<std::size_t>(j)];
                lambda(gj) = (gamma(j) + Scalar{2} * lambda_1_) /
                             (coef_keep(j) * coef_keep(j) +
                              Scalar{2} * lambda_2_);
            }
            alpha_v = (static_cast<Scalar>(n_samples) - gamma.sum() +
                       Scalar{2} * alpha_1_) /
                      (sse + Scalar{2} * alpha_2_);

            // Pruning
            for (Eigen::Index j = 0; j < n_features; ++j) {
                if (lambda(j) >= threshold_lambda_) {
                    keep_lambda(j) = false;
                    coef(j) = Scalar{0};
                } else {
                    keep_lambda(j) = true;
                }
            }

            if (compute_score_) {
                score_log.push_back(objective(
                    static_cast<Scalar>(n_samples), lambda, alpha_v,
                    coef, sse, sigma));
            }

            iter_done = iter;
            if (iter > 0 && (coef_old - coef).cwiseAbs().sum() < tol_) {
                if (verbose_) {
                    std::cout << "Converged after " << iter << " iterations\n";
                }
                break;
            }
            coef_old = coef;

            bool any_keep = false;
            for (Eigen::Index j = 0; j < n_features; ++j) {
                if (keep_lambda(j)) { any_keep = true; break; }
            }
            if (!any_keep) break;
        }
        n_iter_ = iter_done + 1;

        // Final pass with current params if any remain
        bool any_keep = false;
        for (Eigen::Index j = 0; j < n_features; ++j) {
            if (keep_lambda(j)) { any_keep = true; break; }
        }
        if (any_keep) {
            std::vector<Eigen::Index> idx;
            idx.reserve(static_cast<std::size_t>(n_features));
            for (Eigen::Index j = 0; j < n_features; ++j) {
                if (keep_lambda(j)) idx.push_back(j);
            }
            const Eigen::Index k = static_cast<Eigen::Index>(idx.size());
            MatrixType X_keep(n_samples, k);
            VectorType lambda_keep(k);
            for (Eigen::Index j = 0; j < k; ++j) {
                X_keep.col(j) = Xc.col(idx[static_cast<std::size_t>(j)]);
                lambda_keep(j) = lambda(idx[static_cast<std::size_t>(j)]);
            }
            sigma = update_sigma(X_keep, alpha_v, lambda_keep);
            VectorType coef_keep = alpha_v * (sigma * (X_keep.transpose() * yc));
            coef.setZero();
            for (Eigen::Index j = 0; j < k; ++j) {
                coef(idx[static_cast<std::size_t>(j)]) = coef_keep(j);
            }
            sigma_ = sigma;
        } else {
            sigma_ = MatrixType::Zero(0, 0);
        }

        coef_ = coef.transpose();
        alpha_ = alpha_v;
        lambda_vec_ = lambda.transpose();

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

        // Build keep mask for prediction
        keep_mask_ = Eigen::Array<bool, Eigen::Dynamic, 1>(n_features);
        for (Eigen::Index j = 0; j < n_features; ++j) {
            keep_mask_(j) = (lambda(j) < threshold_lambda_);
        }

        this->fitted_ = true;
        return *this;
    }

    /// @brief Fit from a sparse design matrix (densifies internally).
    template <int Options, typename StorageIndex>
    ARDRegression& fit(const Eigen::SparseMatrix<Scalar, Options, StorageIndex>& X,
                       const Eigen::Ref<const VectorType>& y) {
        if (X.rows() == 0 || X.cols() == 0)
            throw std::invalid_argument("ARDRegression.fit: empty sparse matrix.");
        MatrixType Xd = MatrixType(X);
        return fit_impl(Xd, y);
    }

    /// @brief Predictive mean.
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

    /// @brief Predictive mean and standard deviation (tag-dispatch overload).
    [[nodiscard]] std::pair<VectorType, VectorType> predict(
        const Eigen::Ref<const MatrixType>& X, std::true_type) const {
        this->check_is_fitted();
        this->validate_feature_count(X);
        VectorType y_mean = predict_impl(X);

        // Select kept columns (sigma is over kept features only).
        std::vector<Eigen::Index> kept;
        kept.reserve(static_cast<std::size_t>(keep_mask_.size()));
        for (Eigen::Index j = 0; j < keep_mask_.size(); ++j) {
            if (keep_mask_(j)) kept.push_back(j);
        }
        const Eigen::Index k = static_cast<Eigen::Index>(kept.size());
        VectorType y_std(X.rows());
        if (k == 0) {
            y_std = VectorType::Constant(X.rows(),
                                         std::sqrt(Scalar{1} / alpha_));
            return {std::move(y_mean), std::move(y_std)};
        }
        MatrixType X_keep(X.rows(), k);
        for (Eigen::Index j = 0; j < k; ++j) {
            X_keep.col(j) = X.col(kept[static_cast<std::size_t>(j)]);
        }
        MatrixType XS = X_keep * sigma_;
        VectorType sigmas_sq = (XS.array() * X_keep.array()).rowwise().sum();
        y_std = (sigmas_sq.array() + Scalar{1} / alpha_).sqrt();
        return {std::move(y_mean), std::move(y_std)};
    }

    [[nodiscard]] std::pair<VectorType, VectorType> predict(
        const Eigen::Ref<const MatrixType>& X, return_std_t) const {
        return predict(X, std::true_type{});
    }

    using Base::predict;

private:
    static Scalar variance(const Eigen::Ref<const VectorType>& y) {
        if (y.size() == 0) return Scalar{0};
        const Scalar mean = y.mean();
        return (y.array() - mean).square().sum() / static_cast<Scalar>(y.size());
    }

    /// sigma = (diag(lambda_keep) + alpha * X_keep^T X_keep)^{-1}
    static MatrixType update_sigma(const MatrixType& X_keep,
                                   Scalar alpha_v,
                                   const VectorType& lambda_keep) {
        MatrixType gram = X_keep.transpose() * X_keep;
        MatrixType sig_inv = alpha_v * gram;
        sig_inv.diagonal().array() += lambda_keep.array();
        // Symmetric positive-definite: use Cholesky (matches sklearn pinvh
        // for SPD matrices to numerical tolerance).
        Eigen::LLT<MatrixType> llt(sig_inv);
        if (llt.info() == Eigen::Success) {
            MatrixType I = MatrixType::Identity(sig_inv.rows(), sig_inv.cols());
            return llt.solve(I);
        }
        // Fallback: pseudo-inverse via SVD.
        Eigen::JacobiSVD<MatrixType> svd(
            sig_inv, Eigen::ComputeFullU | Eigen::ComputeFullV);
        const Scalar tol_pinv = std::numeric_limits<Scalar>::epsilon() *
                                std::max<Eigen::Index>(sig_inv.rows(),
                                                        sig_inv.cols()) *
                                svd.singularValues().array().abs().maxCoeff();
        VectorType inv_s = svd.singularValues();
        for (Eigen::Index i = 0; i < inv_s.size(); ++i) {
            inv_s(i) = (inv_s(i) > tol_pinv) ? Scalar{1} / inv_s(i)
                                              : Scalar{0};
        }
        return svd.matrixV() * inv_s.asDiagonal() *
               svd.matrixU().transpose();
    }

    Scalar objective(Scalar n_samples,
                     const VectorType& lambda,
                     Scalar alpha_v,
                     const VectorType& coef,
                     Scalar sse,
                     const MatrixType& sigma) const {
        using std::log;
        // fast_logdet via LDLT
        Scalar logdet_sigma;
        Eigen::LDLT<MatrixType> ldlt(sigma);
        if (ldlt.info() == Eigen::Success) {
            VectorType d = ldlt.vectorD();
            Scalar s = Scalar{0};
            bool ok = true;
            for (Eigen::Index i = 0; i < d.size(); ++i) {
                if (d(i) <= Scalar{0}) { ok = false; break; }
                s += log(d(i));
            }
            logdet_sigma = ok ? s
                              : -std::numeric_limits<Scalar>::infinity();
        } else {
            logdet_sigma = -std::numeric_limits<Scalar>::infinity();
        }

        Scalar s = (lambda_1_ * lambda.array().log() -
                    lambda_2_ * lambda.array()).sum();
        s += alpha_1_ * log(alpha_v) - alpha_2_ * alpha_v;
        s += Scalar{0.5} * (
            logdet_sigma
            + n_samples * log(alpha_v)
            + lambda.array().log().sum());
        s -= Scalar{0.5} * (
            alpha_v * sse
            + (lambda.array() * coef.array().square()).sum());
        return s;
    }

    // ---- Hyper-parameters -------------------------------------------------
    int max_iter_;
    Scalar tol_;
    Scalar alpha_1_;
    Scalar alpha_2_;
    Scalar lambda_1_;
    Scalar lambda_2_;
    bool compute_score_;
    Scalar threshold_lambda_;
    bool fit_intercept_;
    bool copy_X_;
    bool verbose_;

    // ---- Fitted attributes ------------------------------------------------
    RowVectorType coef_;
    Scalar intercept_ = Scalar{0};
    Scalar alpha_ = Scalar{0};
    RowVectorType lambda_vec_;
    MatrixType sigma_;
    VectorType scores_;
    int n_iter_ = 0;
    RowVectorType X_offset_;
    RowVectorType X_scale_;
    Scalar y_offset_ = Scalar{0};
    Eigen::Array<bool, Eigen::Dynamic, 1> keep_mask_;
};

/// @}

} // namespace Skigen

#endif // SKIGEN_LINEAR_MODEL_ARD_REGRESSION_H
