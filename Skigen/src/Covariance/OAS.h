// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_COVARIANCE_OAS_H
#define SKIGEN_COVARIANCE_OAS_H

#include "../Core/Base.h"
#include "../Core/Validation.h"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <numbers>

namespace Skigen {

/// @defgroup Algo_OAS OAS
/// @ingroup Covariance
/// @brief Oracle Approximating Shrinkage covariance estimator.
/// @{

/// @brief Oracle Approximating Shrinkage (OAS) covariance estimator.
///
/// Estimates the covariance matrix using the OAS shrinkage formula,
/// which provides an improved estimate over Ledoit-Wolf for Gaussian data.
///
/// The estimator computes:
/// @f[
///   \hat{\Sigma} = (1 - \rho)\,S + \rho\,\mu\,I
/// @f]
/// where the shrinkage coefficient @f$\rho@f$ is given by:
/// @f[
///   \rho = \frac{(1 - 2/p)\,\mathrm{tr}(S^2) + \mathrm{tr}(S)^2}
///               {(n + 1 - 2/p)\,\bigl(\mathrm{tr}(S^2) - \mathrm{tr}(S)^2/p\bigr)}
/// @f]
///
/// Reference: Chen, Wiesel, Eldar & Hero (2010), "Shrinkage Algorithms
/// for MMSE Covariance Estimation", IEEE Trans. Signal Processing,
/// 58(10), 5016-5029.
///
/// Mirrors
/// [sklearn.covariance.OAS](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.OAS.html).
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
/// | `shrinkage()` | `Scalar` | Shrinkage coefficient ρ ∈ [0,1]. |
/// | `location()` | `RowVectorType` | Estimated mean of the data (1 × p). |
///
/// ### Examples
///
/// @snippet covariance.cpp example_oas
template <typename Scalar = double>
class OAS : public Estimator<OAS<Scalar>, Scalar> {
public:
    using Base = Estimator<OAS<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;
    using typename Base::ScalarType;

    /// @brief Construct an OAS estimator.
    ///
    /// @param assume_centered If `true`, data is assumed centered (`bool`, default `false`).
    explicit OAS(bool assume_centered = false)
        : assume_centered_(assume_centered) {}

    // -- Accessors ----------------------------------------------------------

    /// @brief Estimated covariance matrix (p × p).
    [[nodiscard]] const MatrixType& covariance() const {
        this->check_is_fitted();
        return covariance_;
    }

    /// @brief Shrinkage coefficient ρ ∈ [0, 1].
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

    /// @brief Fit the OAS covariance model.
    ///
    /// @param X Data matrix of shape (n_samples, n_features).
    /// @return Reference to the fitted estimator (`*this`).
    OAS& fit(const Eigen::Ref<const MatrixType>& X) {
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

        // Sample covariance (1/n)
        const MatrixType S = (X_centered.transpose() * X_centered)
                             / static_cast<Scalar>(n);

        const Scalar trS = S.trace();
        const Scalar trS2 = S.squaredNorm();  // = tr(S*S) for symmetric S
        const Scalar p_d = static_cast<Scalar>(p);
        const Scalar n_d = static_cast<Scalar>(n);
        const Scalar mu = trS / p_d;

        // OAS shrinkage (Chen et al. 2010, as corrected in scikit-learn):
        //   alpha = mean(S_ij^2) = ||S||_F^2 / p^2,  mu = tr(S) / p
        //   shrinkage = (alpha + mu^2) / [(n + 1) (alpha - mu^2 / p)]
        const Scalar alpha = trS2 / (p_d * p_d);
        const Scalar num = alpha + mu * mu;
        const Scalar den = (n_d + Scalar{1}) * (alpha - mu * mu / p_d);

        shrinkage_ = Scalar{1};
        if (std::abs(den) > Scalar{1e-30}) {
            shrinkage_ = std::clamp(num / den, Scalar{0}, Scalar{1});
        }

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
        const auto n = X.rows();
        const auto p = X.cols();

        MatrixType X_c = assume_centered_
            ? MatrixType(X) : (X.rowwise() - location_).eval();

        MatrixType S_test = (X_c.transpose() * X_c)
                            / static_cast<Scalar>(n);

        Eigen::SelfAdjointEigenSolver<MatrixType> solver(covariance_);
        auto evals = solver.eigenvalues().array().max(Scalar{1e-30}).eval();
        Scalar log_det = evals.log().sum();
        MatrixType cov_inv = solver.eigenvectors()
            * evals.inverse().matrix().asDiagonal()
            * solver.eigenvectors().transpose();
        Scalar tr = (cov_inv * S_test).trace();

        return Scalar{-0.5} * (static_cast<Scalar>(p)
            * std::log(Scalar{2} * std::numbers::pi_v<Scalar>)
            + log_det + tr);
    }

private:
    bool assume_centered_;
    MatrixType covariance_;
    Scalar shrinkage_{0};
    RowVectorType location_;
};

/// @}

} // namespace Skigen

#endif // SKIGEN_COVARIANCE_OAS_H
