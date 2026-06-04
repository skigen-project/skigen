// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_COVARIANCE_LEDOIT_WOLF_H
#define SKIGEN_COVARIANCE_LEDOIT_WOLF_H

#include "../Core/Base.h"
#include "../Core/Validation.h"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <numbers>

namespace Skigen {

/// @defgroup Algo_LedoitWolf Ledoit-Wolf
/// @ingroup Covariance
/// @brief Ledoit-Wolf optimal shrinkage covariance estimator.
/// @{

/// @brief Ledoit-Wolf shrinkage covariance estimator.
///
/// Estimates the covariance matrix using the Ledoit-Wolf shrinkage
/// method, which analytically computes the optimal shrinkage coefficient
/// that minimizes the Frobenius norm of the estimation error.
///
/// The estimator computes:
/// @f[
///   \hat{\Sigma} = (1 - \alpha)\,S + \alpha\,\mu\,I
/// @f]
/// where @f$S@f$ is the sample covariance (normalized by @f$n@f$),
/// @f$\mu = \mathrm{tr}(S)/p@f$ is the mean eigenvalue, and
/// @f$\alpha \in [0,1]@f$ is the optimal shrinkage coefficient.
///
/// Reference: Ledoit & Wolf (2004), "A well-conditioned estimator for
/// large-dimensional covariance matrices", Journal of Multivariate
/// Analysis, 88(2), 365-411.
///
/// Mirrors
/// [sklearn.covariance.LedoitWolf](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html).
///
/// ### Parameters (constructor)
///
/// None — the shrinkage coefficient is determined automatically.
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type | Description |
/// |----------|------|-------------|
/// | `covariance()` | `MatrixType` | Estimated covariance (p × p). |
/// | `shrinkage()` | `Scalar` | Optimal shrinkage coefficient α ∈ [0,1]. |
/// | `location()` | `RowVectorType` | Estimated mean of the data (1 × p). |
///
/// ### Notes
///
/// Input X has shape (n_samples, n_features). This differs from the
/// mne-cpp convention of (n_features, n_samples). The data is centered
/// internally unless `assume_centered` is set.
///
/// ### Limitations relative to scikit-learn `store_precision`, `block_size`
///   are not honoured.
///
/// ### Examples
///
/// @snippet covariance.cpp example_ledoit_wolf
template <typename Scalar = double>
class LedoitWolf
    : public Estimator<LedoitWolf<Scalar>, Scalar> {
public:
    using Base = Estimator<LedoitWolf<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;
    using typename Base::ScalarType;

    /// @brief Construct a LedoitWolf estimator.
    ///
    /// @param assume_centered If `true`, data is assumed to be centered
    ///   and mean is not subtracted (`bool`, default `false`).
    explicit LedoitWolf(bool assume_centered = false)
        : assume_centered_(assume_centered) {}

    // -- Accessors ----------------------------------------------------------

    /// @brief Estimated covariance matrix (p × p).
    [[nodiscard]] const MatrixType& covariance() const {
        this->check_is_fitted();
        return covariance_;
    }

    /// @brief Optimal shrinkage coefficient α ∈ [0, 1].
    [[nodiscard]] Scalar shrinkage() const {
        this->check_is_fitted();
        return shrinkage_;
    }

    /// @brief Estimated location (mean) of the data (1 × p).
    [[nodiscard]] const RowVectorType& location() const {
        this->check_is_fitted();
        return location_;
    }

    SKIGEN_PARAMS((assume_centered, assume_centered_, bool))

    // -- fit ----------------------------------------------------------------

    /// @brief Fit the Ledoit-Wolf covariance model.
    ///
    /// @param X Data matrix of shape (n_samples, n_features).
    /// @return Reference to the fitted estimator (`*this`).
    LedoitWolf& fit(const Eigen::Ref<const MatrixType>& X) {
        internal::check_non_empty(X);
        const auto n = X.rows();
        const auto p = X.cols();
        this->n_features_in_ = p;

        // Center data
        MatrixType X_centered;
        if (assume_centered_) {
            location_ = RowVectorType::Zero(p);
            X_centered = X;
        } else {
            location_ = X.colwise().mean();
            X_centered = X.rowwise() - location_;
        }

        // Sample covariance (1/n normalization, matching sklearn/LW paper)
        // X_centered is (n × p), so S = X^T X / n → (p × p)
        const MatrixType S = (X_centered.transpose() * X_centered)
                             / static_cast<Scalar>(n);

        // Shrinkage target: mu * I_p
        const Scalar mu = S.trace() / static_cast<Scalar>(p);

        // delta = ||S - mu*I||_F^2 / p
        MatrixType diff = S - mu * MatrixType::Identity(p, p);
        const Scalar delta = diff.squaredNorm() / static_cast<Scalar>(p);

        if (delta < Scalar{1e-30}) {
            covariance_ = S;
            shrinkage_ = Scalar{0};
            this->fitted_ = true;
            return *this;
        }

        // beta = sum_k ||x_k x_k^T - S||_F^2 / (n^2 * p)
        // Efficiently: sum_k (x_k^T x_k)^2 - n * ||S||_F^2
        Scalar sum_sq_norms{0};
        for (Eigen::Index k = 0; k < n; ++k) {
            const Scalar nrm = X_centered.row(k).squaredNorm();
            sum_sq_norms += nrm * nrm;
        }
        const Scalar beta_sum = sum_sq_norms
            - static_cast<Scalar>(n) * S.squaredNorm();
        const Scalar beta = beta_sum
            / (static_cast<Scalar>(n) * static_cast<Scalar>(n)
               * static_cast<Scalar>(p));

        // Optimal shrinkage
        shrinkage_ = std::clamp(beta / delta, Scalar{0}, Scalar{1});

        // Shrunk covariance
        covariance_ = (Scalar{1} - shrinkage_) * S;
        covariance_.diagonal().array() += shrinkage_ * mu;

        this->fitted_ = true;
        return *this;
    }

    /// @brief Return the Gaussian log-likelihood of X under the fitted model.
    ///
    /// @param X Data matrix of shape (n_samples, n_features).
    /// @return Average log-likelihood per sample.
    [[nodiscard]] Scalar score(const Eigen::Ref<const MatrixType>& X) const {
        this->check_is_fitted();
        return gaussian_log_likelihood(X, covariance_, location_,
                                       assume_centered_);
    }

private:
    bool assume_centered_;
    MatrixType covariance_;
    Scalar shrinkage_{0};
    RowVectorType location_;

    static Scalar gaussian_log_likelihood(
        const Eigen::Ref<const MatrixType>& X,
        const MatrixType& cov,
        const RowVectorType& mean,
        bool assume_centered) {
        const auto n = X.rows();
        const auto p = X.cols();

        MatrixType X_c = assume_centered ? MatrixType(X)
                                         : (X.rowwise() - mean).eval();

        // S_test = X_c^T X_c / n
        MatrixType S_test = (X_c.transpose() * X_c)
                            / static_cast<Scalar>(n);

        // Eigendecomposition of covariance
        Eigen::SelfAdjointEigenSolver<MatrixType> solver(cov);
        auto evals = solver.eigenvalues().array().max(Scalar{1e-30}).eval();
        auto evecs = solver.eigenvectors();

        Scalar log_det = evals.log().sum();
        MatrixType cov_inv = evecs
            * evals.inverse().matrix().asDiagonal()
            * evecs.transpose();
        Scalar tr_inv_s = (cov_inv * S_test).trace();

        return Scalar{-0.5} * (static_cast<Scalar>(p)
            * std::log(Scalar{2} * std::numbers::pi_v<Scalar>)
            + log_det + tr_inv_s);
    }
};

/// @}

} // namespace Skigen

#endif // SKIGEN_COVARIANCE_LEDOIT_WOLF_H
