// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

#ifndef SKIGEN_KERNEL_RIDGE_KERNEL_RIDGE_H
#define SKIGEN_KERNEL_RIDGE_KERNEL_RIDGE_H

#include "../Core/Base.h"
#include "../Core/Validation.h"
#include "../SVM/Detail/Kernels.h"

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <stdexcept>

namespace Skigen {

/// @defgroup Algo_KernelRidge KernelRidge
/// @ingroup KernelRidge
/// @brief Kernel ridge regression.
/// @{

/// @brief Kernel ridge regression with dense numeric input.
///
/// Fits the dual system
/// @f[
///   (K(X, X) + \alpha I)c = y
/// @f]
/// and predicts with @f$K(X_\text{test}, X_\text{fit})c@f$.
///
/// Mirrors the dense single-target subset of
/// [sklearn.kernel_ridge.KernelRidge](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html).
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `alpha` | `Scalar` | `1` | L2 regularization strength. |
/// | `kernel` | `Kernel` | `Linear` | `Linear`, `RBF`, `Poly`, or `Sigmoid`. |
/// | `gamma` | `Scalar` | `0` | Kernel coefficient; `0` means `1 / n_features`. |
/// | `degree` | `int` | `3` | Polynomial-kernel degree. |
/// | `coef0` | `Scalar` | `1` | Independent term for polynomial and sigmoid kernels. |
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type | Description |
/// |----------|------|-------------|
/// | `dual_coef()` | `VectorType` | Dual coefficients of shape `(n_samples)`. |
/// | `X_fit()` | `MatrixType` | Training samples retained for prediction. |
/// | `gamma_effective()` | `Scalar` | Resolved gamma used by the kernel. |
///
/// ### Limitations relative to scikit-learn
///
/// This first implementation supports dense single-target regression and the
/// shared Skigen kernel set. Multi-output targets, sample weights,
/// precomputed kernels, sparse input, and callable kernels are deferred.
///
/// ### Examples
///
/// @snippet kernel_ridge.cpp example_kernel_ridge
template <typename Scalar = double>
class KernelRidge : public Predictor<KernelRidge<Scalar>, Scalar> {
public:
    using Base = Predictor<KernelRidge<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::IndexType;
    using Kernel = internal::KernelKind;

    explicit KernelRidge(Scalar alpha = Scalar{1},
                         Kernel kernel = Kernel::Linear,
                         Scalar gamma = Scalar{0},
                         int degree = 3,
                         Scalar coef0 = Scalar{1})
        : alpha_(alpha),
          kernel_(kernel),
          gamma_(gamma),
          degree_(degree),
          coef0_(coef0) {}

    [[nodiscard]] Scalar alpha() const noexcept { return alpha_; }
    [[nodiscard]] Kernel kernel() const noexcept { return kernel_; }
    [[nodiscard]] Scalar gamma() const noexcept { return gamma_; }
    [[nodiscard]] int degree() const noexcept { return degree_; }
    [[nodiscard]] Scalar coef0() const noexcept { return coef0_; }

    [[nodiscard]] const VectorType& dual_coef() const {
        this->check_is_fitted();
        return dual_coef_;
    }

    [[nodiscard]] const MatrixType& X_fit() const {
        this->check_is_fitted();
        return X_fit_;
    }

    [[nodiscard]] Scalar gamma_effective() const {
        this->check_is_fitted();
        return gamma_eff_;
    }

    SKIGEN_PARAMS(
        (alpha, alpha_, double),
        (gamma, gamma_, double),
        (degree, degree_, int),
        (coef0, coef0_, double))

    KernelRidge& fit_impl(const Eigen::Ref<const MatrixType>& X,
                          const Eigen::Ref<const VectorType>& y) {
        internal::check_non_empty(X);
        internal::check_consistent_length(X, y);
        if (alpha_ < Scalar{0}) {
            throw std::invalid_argument("KernelRidge: alpha must be non-negative.");
        }
        if (degree_ < 0) {
            throw std::invalid_argument("KernelRidge: degree must be non-negative.");
        }

        this->n_features_in_ = X.cols();
        gamma_eff_ = gamma_ > Scalar{0}
            ? gamma_
            : Scalar{1} / static_cast<Scalar>(X.cols());
        X_fit_ = X;

        MatrixType gram = kernel_matrix(X_fit_, X_fit_);
        gram.diagonal().array() += alpha_;
        dual_coef_ = gram.ldlt().solve(y);

        this->fitted_ = true;
        return *this;
    }

    [[nodiscard]] VectorType predict_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        VectorType predictions(X.rows());
        for (IndexType row = 0; row < X.rows(); ++row) {
            Scalar value = Scalar{0};
            for (IndexType train = 0; train < X_fit_.rows(); ++train) {
                value += dual_coef_(train) * internal::kernel_eval<Scalar>(
                    kernel_, X.row(row), X_fit_.row(train), gamma_eff_, coef0_, degree_);
            }
            predictions(row) = value;
        }
        return predictions;
    }

    [[nodiscard]] Scalar score_impl(const Eigen::Ref<const MatrixType>& X,
                                    const Eigen::Ref<const VectorType>& y) const {
        internal::check_consistent_length(X, y);
        const VectorType predictions = predict_impl(X);
        const Scalar ss_res = (y - predictions).squaredNorm();
        const Scalar mean = y.mean();
        const Scalar ss_tot = (y.array() - mean).square().sum();
        return ss_tot > Scalar{0} ? Scalar{1} - ss_res / ss_tot : Scalar{0};
    }

private:
    [[nodiscard]] MatrixType kernel_matrix(const MatrixType& A, const MatrixType& B) const {
        MatrixType out(A.rows(), B.rows());
        for (IndexType i = 0; i < A.rows(); ++i) {
            for (IndexType j = 0; j < B.rows(); ++j) {
                out(i, j) = internal::kernel_eval<Scalar>(
                    kernel_, A.row(i), B.row(j), gamma_eff_, coef0_, degree_);
            }
        }
        return out;
    }

    Scalar alpha_;
    Kernel kernel_;
    Scalar gamma_;
    int degree_;
    Scalar coef0_;
    Scalar gamma_eff_ = Scalar{0};
    MatrixType X_fit_;
    VectorType dual_coef_;
};

/// @}

}  // namespace Skigen

#endif  // SKIGEN_KERNEL_RIDGE_KERNEL_RIDGE_H
