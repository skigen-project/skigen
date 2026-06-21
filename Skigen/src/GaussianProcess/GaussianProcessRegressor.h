// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

#ifndef SKIGEN_GAUSSIAN_PROCESS_REGRESSOR_H
#define SKIGEN_GAUSSIAN_PROCESS_REGRESSOR_H

#include "../Core/Base.h"
#include "../Core/Validation.h"

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace Skigen {

/// @defgroup Algo_GaussianProcess Gaussian Process
/// @ingroup GaussianProcess
/// @brief Dense Gaussian process regression.
/// @{

namespace gaussian_process {

/// @brief Built-in dense covariance kernels for GaussianProcessRegressor.
///
/// The enum mirrors the highest-value sklearn.gaussian_process.kernels
/// primitives while keeping Skigen dependency-free and header-only.
enum class Kernel {
    RBF,
    Matern,
    RationalQuadratic,
    ExpSineSquared,
    DotProduct,
    White,
    Constant
};

}  // namespace gaussian_process

/// @brief Dense single-target Gaussian Process regressor.
///
/// Fits the Cholesky system
/// @f[
///   (K(X, X) + \alpha I) a = y
/// @f]
/// and predicts posterior means and optional posterior uncertainty.
///
/// Mirrors the dense, fixed-hyperparameter subset of
/// [sklearn.gaussian_process.GaussianProcessRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html).
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `kernel` | `Kernel` | `RBF` | Built-in covariance kernel. |
/// | `alpha` | `Scalar` | `1e-10` | Diagonal jitter / observation noise variance. |
/// | `length_scale` | `Scalar` | `1` | Isotropic length scale for stationary kernels. |
/// | `constant_value` | `Scalar` | `1` | Amplitude for stationary kernels or value for `Constant`. |
/// | `noise_level` | `Scalar` | `1` | Diagonal value for `White`. |
/// | `nu` | `Scalar` | `1.5` | Matern smoothness; supports `0.5`, `1.5`, and `2.5`. |
/// | `rational_quadratic_alpha` | `Scalar` | `1` | Shape parameter for `RationalQuadratic`. |
/// | `periodicity` | `Scalar` | `1` | Period for `ExpSineSquared`. |
/// | `sigma_0` | `Scalar` | `1` | Bias scale for `DotProduct`. |
/// | `normalize_y` | `bool` | `false` | Center targets during fitting. |
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type | Description |
/// |----------|------|-------------|
/// | `X_train()` | `MatrixType` | Training samples retained for prediction. |
/// | `dual_coef()` | `VectorType` | Cholesky-solved dual coefficients. |
/// | `L()` | `MatrixType` | Lower Cholesky factor of the regularized Gram matrix. |
/// | `y_train_mean()` | `Scalar` | Target mean subtracted when `normalize_y=true`. |
/// | `log_marginal_likelihood_value()` | `Scalar` | Log marginal likelihood at fixed hyperparameters. |
///
/// ### Limitations relative to scikit-learn
///
/// Hyperparameter optimization, callable/composite kernels, multi-output
/// targets, sample weights, and sparse input are deferred. Kernel sums/products
/// can be emulated in a later release by extending the kernel policy layer.
///
/// ### Examples
///
/// @snippet gaussian_process_regressor.cpp example_gaussian_process_regressor
template <typename Scalar = double>
class GaussianProcessRegressor
    : public Predictor<GaussianProcessRegressor<Scalar>, Scalar> {
public:
    using Base = Predictor<GaussianProcessRegressor<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::IndexType;
    using Kernel = gaussian_process::Kernel;

    explicit GaussianProcessRegressor(
        Kernel kernel = Kernel::RBF,
        Scalar alpha = Scalar{1e-10},
        Scalar length_scale = Scalar{1},
        Scalar constant_value = Scalar{1},
        Scalar noise_level = Scalar{1},
        Scalar nu = Scalar{1.5},
        Scalar rational_quadratic_alpha = Scalar{1},
        Scalar periodicity = Scalar{1},
        Scalar sigma_0 = Scalar{1},
        bool normalize_y = false)
        : kernel_(kernel),
          alpha_(alpha),
          length_scale_(length_scale),
          constant_value_(constant_value),
          noise_level_(noise_level),
          nu_(nu),
          rational_quadratic_alpha_(rational_quadratic_alpha),
          periodicity_(periodicity),
          sigma_0_(sigma_0),
          normalize_y_(normalize_y) {}

    [[nodiscard]] Kernel kernel() const noexcept { return kernel_; }
    [[nodiscard]] Scalar alpha() const noexcept { return alpha_; }
    [[nodiscard]] Scalar length_scale() const noexcept { return length_scale_; }
    [[nodiscard]] Scalar constant_value() const noexcept { return constant_value_; }
    [[nodiscard]] Scalar noise_level() const noexcept { return noise_level_; }
    [[nodiscard]] Scalar nu() const noexcept { return nu_; }
    [[nodiscard]] Scalar rational_quadratic_alpha() const noexcept {
        return rational_quadratic_alpha_;
    }
    [[nodiscard]] Scalar periodicity() const noexcept { return periodicity_; }
    [[nodiscard]] Scalar sigma_0() const noexcept { return sigma_0_; }
    [[nodiscard]] bool normalize_y() const noexcept { return normalize_y_; }

    [[nodiscard]] const MatrixType& X_train() const {
        this->check_is_fitted();
        return X_train_;
    }

    [[nodiscard]] const VectorType& dual_coef() const {
        this->check_is_fitted();
        return dual_coef_;
    }

    [[nodiscard]] const MatrixType& L() const {
        this->check_is_fitted();
        return L_;
    }

    [[nodiscard]] Scalar y_train_mean() const {
        this->check_is_fitted();
        return y_train_mean_;
    }

    [[nodiscard]] Scalar log_marginal_likelihood_value() const {
        this->check_is_fitted();
        return log_marginal_likelihood_;
    }

    SKIGEN_PARAMS(
        (alpha, alpha_, double),
        (length_scale, length_scale_, double),
        (constant_value, constant_value_, double),
        (noise_level, noise_level_, double),
        (nu, nu_, double),
        (rational_quadratic_alpha, rational_quadratic_alpha_, double),
        (periodicity, periodicity_, double),
        (sigma_0, sigma_0_, double),
        (normalize_y, normalize_y_, bool))

    GaussianProcessRegressor& fit_impl(const Eigen::Ref<const MatrixType>& X,
                                       const Eigen::Ref<const VectorType>& y) {
        internal::check_non_empty(X);
        internal::check_consistent_length(X, y);
        validate_parameters();

        this->n_features_in_ = X.cols();
        X_train_ = X;
        y_train_mean_ = normalize_y_ ? y.mean() : Scalar{0};
        const VectorType y_centered = (y.array() - y_train_mean_).matrix();

        MatrixType gram = kernel_matrix(X_train_, X_train_, /*same_matrix=*/true);
        gram.diagonal().array() += alpha_;

        Eigen::LLT<MatrixType> llt;
        Scalar jitter = std::max(alpha_, Scalar{1e-12});
        bool success = false;
        for (int attempt = 0; attempt < 6; ++attempt) {
            llt.compute(gram);
            if (llt.info() == Eigen::Success) {
                success = true;
                break;
            }
            gram.diagonal().array() += jitter;
            jitter *= Scalar{10};
        }
        if (!success) {
            throw std::runtime_error(
                "GaussianProcessRegressor: Cholesky factorization failed; "
                "increase alpha or choose a positive-definite kernel.");
        }

        L_ = llt.matrixL();
        dual_coef_ = llt.solve(y_centered);
        if (llt.info() != Eigen::Success) {
            throw std::runtime_error("GaussianProcessRegressor: linear solve failed.");
        }

        log_marginal_likelihood_ =
            -Scalar{0.5} * y_centered.dot(dual_coef_) -
            L_.diagonal().array().log().sum() -
            Scalar{0.5} * static_cast<Scalar>(X.rows()) *
                std::log(Scalar{2} * pi());

        this->fitted_ = true;
        return *this;
    }

    [[nodiscard]] VectorType predict_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        const MatrixType k_trans = kernel_matrix(X, X_train_, /*same_matrix=*/false);
        return (k_trans * dual_coef_).array() + y_train_mean_;
    }

    [[nodiscard]] VectorType predict_std(
        const Eigen::Ref<const MatrixType>& X) const {
        this->check_is_fitted();
        this->validate_feature_count(X);
        const MatrixType covariance = predict_covariance(X);
        VectorType out(covariance.rows());
        for (IndexType i = 0; i < covariance.rows(); ++i) {
            out(i) = std::sqrt(std::max(Scalar{0}, covariance(i, i)));
        }
        return out;
    }

    [[nodiscard]] MatrixType predict_covariance(
        const Eigen::Ref<const MatrixType>& X) const {
        this->check_is_fitted();
        this->validate_feature_count(X);

        const MatrixType k_trans = kernel_matrix(X, X_train_, /*same_matrix=*/false);
        const MatrixType v = L_.template triangularView<Eigen::Lower>()
                                 .solve(k_trans.transpose());
        MatrixType covariance = kernel_matrix(X, X, /*same_matrix=*/true) -
                                v.transpose() * v;
        covariance = Scalar{0.5} * (covariance + covariance.transpose()).eval();
        for (IndexType i = 0; i < covariance.rows(); ++i) {
            covariance(i, i) = std::max(Scalar{0}, covariance(i, i));
        }
        return covariance;
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
    static constexpr Scalar pi() noexcept {
        return Scalar{3.141592653589793238462643383279502884L};
    }

    static bool approx_equal(Scalar a, Scalar b) noexcept {
        return std::abs(a - b) <= Scalar{1e-12};
    }

    void validate_parameters() const {
        if (alpha_ < Scalar{0}) {
            throw std::invalid_argument(
                "GaussianProcessRegressor: alpha must be non-negative.");
        }
        if (length_scale_ <= Scalar{0}) {
            throw std::invalid_argument(
                "GaussianProcessRegressor: length_scale must be positive.");
        }
        if (constant_value_ < Scalar{0}) {
            throw std::invalid_argument(
                "GaussianProcessRegressor: constant_value must be non-negative.");
        }
        if (noise_level_ < Scalar{0}) {
            throw std::invalid_argument(
                "GaussianProcessRegressor: noise_level must be non-negative.");
        }
        if (rational_quadratic_alpha_ <= Scalar{0}) {
            throw std::invalid_argument(
                "GaussianProcessRegressor: rational_quadratic_alpha must be positive.");
        }
        if (periodicity_ <= Scalar{0}) {
            throw std::invalid_argument(
                "GaussianProcessRegressor: periodicity must be positive.");
        }
        if (sigma_0_ < Scalar{0}) {
            throw std::invalid_argument(
                "GaussianProcessRegressor: sigma_0 must be non-negative.");
        }
        if (kernel_ == Kernel::Matern &&
            !(approx_equal(nu_, Scalar{0.5}) || approx_equal(nu_, Scalar{1.5}) ||
              approx_equal(nu_, Scalar{2.5}))) {
            throw std::invalid_argument(
                "GaussianProcessRegressor: Matern nu must be 0.5, 1.5, or 2.5.");
        }
    }

    [[nodiscard]] MatrixType kernel_matrix(const Eigen::Ref<const MatrixType>& A,
                                           const Eigen::Ref<const MatrixType>& B,
                                           bool same_matrix) const {
        MatrixType out(A.rows(), B.rows());
        for (IndexType i = 0; i < A.rows(); ++i) {
            for (IndexType j = 0; j < B.rows(); ++j) {
                out(i, j) = kernel_value(A.row(i), B.row(j), same_matrix && i == j);
            }
        }
        return out;
    }

    template <typename DerivedA, typename DerivedB>
    [[nodiscard]] Scalar kernel_value(const Eigen::MatrixBase<DerivedA>& a,
                                      const Eigen::MatrixBase<DerivedB>& b,
                                      bool same_observation) const {
        const Scalar distance = (a - b).norm();
        const Scalar d2 = distance * distance;
        const Scalar length2 = length_scale_ * length_scale_;

        switch (kernel_) {
            case Kernel::RBF:
                return constant_value_ * std::exp(-Scalar{0.5} * d2 / length2);
            case Kernel::Matern:
                return constant_value_ * matern_value(distance);
            case Kernel::RationalQuadratic:
                return constant_value_ * std::pow(
                    Scalar{1} + d2 / (Scalar{2} * rational_quadratic_alpha_ * length2),
                    -rational_quadratic_alpha_);
            case Kernel::ExpSineSquared: {
                const Scalar s = std::sin(pi() * distance / periodicity_);
                return constant_value_ * std::exp(-Scalar{2} * s * s / length2);
            }
            case Kernel::DotProduct:
                return sigma_0_ * sigma_0_ + a.dot(b);
            case Kernel::White:
                return same_observation ? noise_level_ : Scalar{0};
            case Kernel::Constant:
                return constant_value_;
        }
        return Scalar{0};
    }

    [[nodiscard]] Scalar matern_value(Scalar distance) const {
        const Scalar scaled = distance / length_scale_;
        if (approx_equal(nu_, Scalar{0.5})) {
            return std::exp(-scaled);
        }
        if (approx_equal(nu_, Scalar{1.5})) {
            const Scalar r = std::sqrt(Scalar{3}) * scaled;
            return (Scalar{1} + r) * std::exp(-r);
        }
        const Scalar r = std::sqrt(Scalar{5}) * scaled;
        return (Scalar{1} + r + r * r / Scalar{3}) * std::exp(-r);
    }

    Kernel kernel_;
    Scalar alpha_;
    Scalar length_scale_;
    Scalar constant_value_;
    Scalar noise_level_;
    Scalar nu_;
    Scalar rational_quadratic_alpha_;
    Scalar periodicity_;
    Scalar sigma_0_;
    bool normalize_y_;

    MatrixType X_train_;
    MatrixType L_;
    VectorType dual_coef_;
    Scalar y_train_mean_ = Scalar{0};
    Scalar log_marginal_likelihood_ = Scalar{0};
};

/// @}

}  // namespace Skigen

#endif  // SKIGEN_GAUSSIAN_PROCESS_REGRESSOR_H
