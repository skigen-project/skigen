// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

#ifndef SKIGEN_GAUSSIAN_PROCESS_KERNELS_H
#define SKIGEN_GAUSSIAN_PROCESS_KERNELS_H

#include <Eigen/Core>

#include <cmath>
#include <stdexcept>
#include <string>

namespace Skigen::gaussian_process {

/// @brief Built-in dense covariance kernels for Gaussian Process estimators.
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

namespace detail {

template <typename Scalar>
constexpr Scalar pi() noexcept {
    return Scalar{3.141592653589793238462643383279502884L};
}

template <typename Scalar>
bool approx_equal(Scalar a, Scalar b) noexcept {
    return std::abs(a - b) <= Scalar{1e-12};
}

template <typename Scalar>
void validate_kernel_parameters(Kernel kernel,
                                Scalar alpha,
                                Scalar length_scale,
                                Scalar constant_value,
                                Scalar noise_level,
                                Scalar nu,
                                Scalar rational_quadratic_alpha,
                                Scalar periodicity,
                                Scalar sigma_0,
                                const char* estimator_name) {
    if (alpha < Scalar{0}) {
        throw std::invalid_argument(
            std::string(estimator_name) + ": alpha must be non-negative.");
    }
    if (length_scale <= Scalar{0}) {
        throw std::invalid_argument(
            std::string(estimator_name) + ": length_scale must be positive.");
    }
    if (constant_value < Scalar{0}) {
        throw std::invalid_argument(
            std::string(estimator_name) + ": constant_value must be non-negative.");
    }
    if (noise_level < Scalar{0}) {
        throw std::invalid_argument(
            std::string(estimator_name) + ": noise_level must be non-negative.");
    }
    if (rational_quadratic_alpha <= Scalar{0}) {
        throw std::invalid_argument(
            std::string(estimator_name) + ": rational_quadratic_alpha must be positive.");
    }
    if (periodicity <= Scalar{0}) {
        throw std::invalid_argument(
            std::string(estimator_name) + ": periodicity must be positive.");
    }
    if (sigma_0 < Scalar{0}) {
        throw std::invalid_argument(
            std::string(estimator_name) + ": sigma_0 must be non-negative.");
    }
    if (kernel == Kernel::Matern &&
        !(approx_equal(nu, Scalar{0.5}) || approx_equal(nu, Scalar{1.5}) ||
          approx_equal(nu, Scalar{2.5}))) {
        throw std::invalid_argument(
            std::string(estimator_name) + ": Matern nu must be 0.5, 1.5, or 2.5.");
    }
}

template <typename Scalar>
Scalar matern_value(Scalar distance, Scalar length_scale, Scalar nu) {
    const Scalar scaled = distance / length_scale;
    if (approx_equal(nu, Scalar{0.5})) {
        return std::exp(-scaled);
    }
    if (approx_equal(nu, Scalar{1.5})) {
        const Scalar r = std::sqrt(Scalar{3}) * scaled;
        return (Scalar{1} + r) * std::exp(-r);
    }
    const Scalar r = std::sqrt(Scalar{5}) * scaled;
    return (Scalar{1} + r + r * r / Scalar{3}) * std::exp(-r);
}

template <typename Scalar, typename DerivedA, typename DerivedB>
Scalar kernel_value(Kernel kernel,
                    const Eigen::MatrixBase<DerivedA>& a,
                    const Eigen::MatrixBase<DerivedB>& b,
                    bool same_observation,
                    Scalar length_scale,
                    Scalar constant_value,
                    Scalar noise_level,
                    Scalar nu,
                    Scalar rational_quadratic_alpha,
                    Scalar periodicity,
                    Scalar sigma_0) {
    const Scalar distance = (a - b).norm();
    const Scalar d2 = distance * distance;
    const Scalar length2 = length_scale * length_scale;

    switch (kernel) {
        case Kernel::RBF:
            return constant_value * std::exp(-Scalar{0.5} * d2 / length2);
        case Kernel::Matern:
            return constant_value * matern_value(distance, length_scale, nu);
        case Kernel::RationalQuadratic:
            return constant_value * std::pow(
                Scalar{1} + d2 / (Scalar{2} * rational_quadratic_alpha * length2),
                -rational_quadratic_alpha);
        case Kernel::ExpSineSquared: {
            const Scalar s = std::sin(pi<Scalar>() * distance / periodicity);
            return constant_value * std::exp(-Scalar{2} * s * s / length2);
        }
        case Kernel::DotProduct:
            return sigma_0 * sigma_0 + a.dot(b);
        case Kernel::White:
            return same_observation ? noise_level : Scalar{0};
        case Kernel::Constant:
            return constant_value;
    }
    return Scalar{0};
}

template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> kernel_matrix(
    Kernel kernel,
    const Eigen::Ref<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>& A,
    const Eigen::Ref<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>& B,
    bool same_matrix,
    Scalar length_scale,
    Scalar constant_value,
    Scalar noise_level,
    Scalar nu,
    Scalar rational_quadratic_alpha,
    Scalar periodicity,
    Scalar sigma_0) {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> out(A.rows(), B.rows());
    for (Eigen::Index i = 0; i < A.rows(); ++i) {
        for (Eigen::Index j = 0; j < B.rows(); ++j) {
            out(i, j) = kernel_value(
                kernel,
                A.row(i),
                B.row(j),
                same_matrix && i == j,
                length_scale,
                constant_value,
                noise_level,
                nu,
                rational_quadratic_alpha,
                periodicity,
                sigma_0);
        }
    }
    return out;
}

}  // namespace detail

}  // namespace Skigen::gaussian_process

#endif  // SKIGEN_GAUSSIAN_PROCESS_KERNELS_H
