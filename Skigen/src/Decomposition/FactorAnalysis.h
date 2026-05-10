// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_DECOMPOSITION_FACTOR_ANALYSIS_H
#define SKIGEN_DECOMPOSITION_FACTOR_ANALYSIS_H

#include "../Core/Base.h"
#include "../Core/Validation.h"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>

namespace Skigen {

/// @defgroup Algo_FactorAnalysis FactorAnalysis
/// @ingroup Decomposition
/// @brief Factor Analysis with EM algorithm.
/// @{

/// @brief Factor Analysis estimator.
///
/// Estimates a covariance model of the form:
/// @f[
///   \Sigma = W W^\top + \mathrm{diag}(\psi)
/// @f]
/// where @f$W@f$ is the loading matrix (n_features × n_components) and
/// @f$\psi@f$ are the per-feature noise variances.
///
/// Uses the Expectation-Maximization (EM) algorithm:
/// - **E-step:** Compute whitened covariance @f$\Psi^{-1/2} S \Psi^{-1/2}@f$,
///   eigendecompose, and extract top-k components.
/// - **M-step:** Update @f$W = \Psi^{1/2} V_k (\Lambda_k - I)^{1/2}@f$
///   and @f$\psi = \mathrm{diag}(S - W W^\top)@f$.
///
/// Mirrors
/// [sklearn.decomposition.FactorAnalysis](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FactorAnalysis.html).
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `n_components` | `int` | 0 (auto) | Number of latent factors. 0 = `min(n,p)/2`. |
/// | `max_iter` | `int` | 1000 | Maximum EM iterations. |
/// | `tol` | `Scalar` | 1e-3 | Convergence tolerance on log-likelihood. |
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type | Description |
/// |----------|------|-------------|
/// | `components()` | `MatrixType` | Loading matrix W (p × k). |
/// | `noise_variance()` | `VectorType` | Per-feature noise variance ψ (p). |
/// | `covariance()` | `MatrixType` | Estimated covariance W W^T + diag(ψ) (p × p). |
/// | `log_likelihood()` | `Scalar` | Final log-likelihood. |
/// | `n_iter()` | `int` | Number of EM iterations run. |
///
/// ### Limitations relative to scikit-learn `svd_method`, `rotation`,
///   `random_state` are not honoured.
///
/// ### Examples
///
/// @snippet decomposition.cpp example_factor_analysis
template <typename Scalar = double>
class FactorAnalysis
    : public Estimator<FactorAnalysis<Scalar>, Scalar> {
public:
    using Base = Estimator<FactorAnalysis<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;
    using typename Base::ScalarType;

    /// @brief Construct a FactorAnalysis estimator.
    ///
    /// @param n_components Number of latent factors (0 = auto).
    /// @param max_iter Maximum EM iterations.
    /// @param tol Convergence tolerance on log-likelihood change.
    explicit FactorAnalysis(int n_components = 0,
                            int max_iter = 1000,
                            Scalar tol = Scalar{1e-3})
        : n_components_(n_components)
        , max_iter_(max_iter)
        , tol_(tol) {}

    // -- Accessors ----------------------------------------------------------

    /// @brief Loading matrix W of shape (n_features, n_components).
    [[nodiscard]] const MatrixType& components() const {
        this->check_is_fitted();
        return components_;
    }

    /// @brief Per-feature noise variances ψ of length n_features.
    [[nodiscard]] const VectorType& noise_variance() const {
        this->check_is_fitted();
        return noise_variance_;
    }

    /// @brief Estimated covariance matrix (n_features × n_features).
    ///
    /// Computed as @f$ W W^\top + \mathrm{diag}(\psi) @f$.
    [[nodiscard]] MatrixType covariance() const {
        this->check_is_fitted();
        MatrixType cov = components_ * components_.transpose();
        cov.diagonal() += noise_variance_;
        return cov;
    }

    /// @brief Final log-likelihood of the fitted model.
    [[nodiscard]] Scalar log_likelihood() const {
        this->check_is_fitted();
        return log_likelihood_;
    }

    /// @brief Number of EM iterations performed.
    [[nodiscard]] int n_iter() const {
        this->check_is_fitted();
        return n_iter_;
    }

    // -- fit ----------------------------------------------------------------

    /// @brief Fit the factor analysis model via EM.
    ///
    /// @param X Data matrix of shape (n_samples, n_features).
    /// @return Reference to the fitted estimator (`*this`).
    FactorAnalysis& fit(const Eigen::Ref<const MatrixType>& X) {
        internal::check_non_empty(X);
        const auto n = X.rows();
        const auto p = X.cols();
        this->n_features_in_ = p;

        // Center data
        mean_ = X.colwise().mean();
        const MatrixType X_c = X.rowwise() - mean_;

        // Sample covariance (1/n normalization)
        const MatrixType S = (X_c.transpose() * X_c)
                             / static_cast<Scalar>(n);

        // Determine number of components
        int k = n_components_;
        if (k <= 0) {
            k = std::max(1, static_cast<int>(std::min(n, p)) / 2);
        }
        k = std::min(k, static_cast<int>(p) - 1);

        // Initialize noise variances from diagonal of S
        VectorType psi = S.diagonal();

        constexpr Scalar eps = Scalar{1e-30};
        Scalar prev_ll = -std::numeric_limits<Scalar>::infinity();
        n_iter_ = 0;

        for (int iter = 0; iter < max_iter_; ++iter) {
            // E-step: whitened covariance Psi^{-1/2} S Psi^{-1/2}
            const VectorType psi_inv_sqrt =
                psi.array().max(eps).inverse().sqrt();

            MatrixType Sw = psi_inv_sqrt.asDiagonal() * S
                          * psi_inv_sqrt.asDiagonal();

            // Eigendecomposition (ascending order)
            Eigen::SelfAdjointEigenSolver<MatrixType> solver(Sw);
            const VectorType& evals = solver.eigenvalues();
            const MatrixType& evecs = solver.eigenvectors();

            // Top-k eigenvalues/vectors (last k in ascending order)
            const VectorType top_evals = evals.tail(k);
            const MatrixType top_evecs = evecs.rightCols(k);

            // Loading matrix: W = Psi^{1/2} * V_k * (Lambda_k - I)^{1/2}
            VectorType load_scale(k);
            for (int j = 0; j < k; ++j) {
                load_scale(j) = std::sqrt(
                    std::max(top_evals(j) - Scalar{1}, Scalar{0}));
            }
            components_ = psi.array().sqrt().matrix().asDiagonal()
                        * top_evecs * load_scale.asDiagonal();

            // M-step: update noise variances
            const MatrixType WWT = components_ * components_.transpose();
            psi = (S - WWT).diagonal().array().max(eps);

            // Log-likelihood via eigendecomposition of Sigma = WW^T+diag(psi)
            MatrixType Sigma = WWT;
            Sigma.diagonal() += psi;

            Eigen::SelfAdjointEigenSolver<MatrixType> sig_solver(Sigma);
            const VectorType sig_evals =
                sig_solver.eigenvalues().array().max(eps);

            const Scalar log_det = sig_evals.array().log().sum();

            const MatrixType Sigma_inv =
                sig_solver.eigenvectors()
                * sig_evals.array().inverse().matrix().asDiagonal()
                * sig_solver.eigenvectors().transpose();

            const Scalar tr_inv_S = (Sigma_inv * S).trace();

            const Scalar ll = Scalar{-0.5} * static_cast<Scalar>(n)
                * (static_cast<Scalar>(p)
                       * std::log(Scalar{2} * std::numbers::pi_v<Scalar>)
                   + log_det + tr_inv_S);

            n_iter_ = iter + 1;

            if (std::abs(ll - prev_ll) < tol_) {
                log_likelihood_ = ll;
                break;
            }
            prev_ll = ll;
            log_likelihood_ = ll;
        }

        noise_variance_ = psi;
        this->fitted_ = true;
        return *this;
    }

    /// @brief Return the Gaussian log-likelihood of X under the fitted model.
    ///
    /// @param X Data matrix of shape (n_samples, n_features).
    /// @return Average log-likelihood per sample.
    [[nodiscard]] Scalar score(const Eigen::Ref<const MatrixType>& X) const {
        this->check_is_fitted();
        const auto n = X.rows();
        const auto p = X.cols();

        const MatrixType X_c = (X.rowwise() - mean_);
        const MatrixType S_test =
            (X_c.transpose() * X_c) / static_cast<Scalar>(n);

        const MatrixType cov = covariance();
        Eigen::SelfAdjointEigenSolver<MatrixType> solver(cov);
        const VectorType evals =
            solver.eigenvalues().array().max(Scalar{1e-30});

        const Scalar log_det = evals.array().log().sum();
        const MatrixType cov_inv =
            solver.eigenvectors()
            * evals.array().inverse().matrix().asDiagonal()
            * solver.eigenvectors().transpose();
        const Scalar tr = (cov_inv * S_test).trace();

        return Scalar{-0.5}
            * (static_cast<Scalar>(p)
                   * std::log(Scalar{2} * std::numbers::pi_v<Scalar>)
               + log_det + tr);
    }

private:
    int n_components_;
    int max_iter_;
    Scalar tol_;

    MatrixType components_;       // W: (p × k)
    VectorType noise_variance_;   // ψ: (p)
    RowVectorType mean_;          // (1 × p)
    Scalar log_likelihood_{0};
    int n_iter_{0};
};

/// @}

}  // namespace Skigen

#endif  // SKIGEN_DECOMPOSITION_FACTOR_ANALYSIS_H
