// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_COVARIANCE_EMPIRICAL_COVARIANCE_H
#define SKIGEN_COVARIANCE_EMPIRICAL_COVARIANCE_H

#include "../Core/Base.h"
#include "../Core/Validation.h"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <numbers>

namespace Skigen {

/// @defgroup Algo_EmpiricalCovariance Empirical Covariance
/// @ingroup Covariance
/// @brief Maximum-likelihood covariance estimator.
/// @{

/// @brief Maximum likelihood covariance estimator.
///
/// Computes the sample covariance matrix from data.
///
/// Mirrors
/// [sklearn.covariance.EmpiricalCovariance](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EmpiricalCovariance.html).
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `assume_centered` | `bool` | `false` | If `true`, data is assumed to be centered. |
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type | Description |
/// |----------|------|-------------|
/// | `covariance()` | `MatrixType` | Estimated covariance (p × p). |
/// | `location()` | `RowVectorType` | Estimated mean (1 × p). |
///
/// ### Examples
///
/// @snippet covariance.cpp example_empirical_covariance
template <typename Scalar = double>
class EmpiricalCovariance
    : public Estimator<EmpiricalCovariance<Scalar>, Scalar> {
public:
    using Base = Estimator<EmpiricalCovariance<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;
    using typename Base::ScalarType;

    /// @brief Construct an EmpiricalCovariance estimator.
    explicit EmpiricalCovariance(bool assume_centered = false)
        : assume_centered_(assume_centered) {}

    // -- Accessors ----------------------------------------------------------

    /// @brief Estimated covariance matrix (p × p).
    [[nodiscard]] const MatrixType& covariance() const {
        this->check_is_fitted();
        return covariance_;
    }

    /// @brief Estimated location (mean) of the data (1 × p).
    [[nodiscard]] const RowVectorType& location() const {
        this->check_is_fitted();
        return location_;
    }

    // -- fit ----------------------------------------------------------------

    /// @brief Fit the empirical covariance model.
    ///
    /// @param X Data matrix of shape (n_samples, n_features).
    /// @return Reference to the fitted estimator (`*this`).
    EmpiricalCovariance& fit(const Eigen::Ref<const MatrixType>& X) {
        internal::check_non_empty(X);
        const auto n = X.rows();
        const auto p = X.cols();
        this->n_features_in_ = p;

        MatrixType X_c;
        if (assume_centered_) {
            location_ = RowVectorType::Zero(p);
            X_c = X;
        } else {
            location_ = X.colwise().mean();
            X_c = X.rowwise() - location_;
        }

        covariance_ = (X_c.transpose() * X_c)
                       / static_cast<Scalar>(n);

        this->fitted_ = true;
        return *this;
    }

    /// @brief Return Gaussian log-likelihood of X under the fitted model.
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
    RowVectorType location_;
};

/// @}

} // namespace Skigen

#endif // SKIGEN_COVARIANCE_EMPIRICAL_COVARIANCE_H
