// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_CORE_EIGEN_HELPERS_H
#define SKIGEN_CORE_EIGEN_HELPERS_H

#include <Eigen/Core>
#include <cmath>
#include <limits>

namespace Skigen {
namespace internal {

// ---------------------------------------------------------------------------
// Handle near-zero values in scale vectors.
// Mirrors scikit-learn's _handle_zeros_in_scale():
//   values < 10 * eps are replaced with 1.0
// ---------------------------------------------------------------------------

template <typename Derived>
void handle_zeros_in_scale(Eigen::MatrixBase<Derived>& scale) {
    using Scalar = typename Derived::Scalar;
    constexpr Scalar threshold =
        Scalar{10} * std::numeric_limits<Scalar>::epsilon();
    scale.derived() = (scale.array() < threshold)
                          .select(Derived::Ones(scale.rows(), scale.cols()),
                                  scale.derived());
}

// ---------------------------------------------------------------------------
// Column-wise variance (biased, ddof=0) matching numpy/scikit-learn convention
// ---------------------------------------------------------------------------

template <typename Scalar>
Eigen::Matrix<Scalar, 1, Eigen::Dynamic> colwise_variance(
    const Eigen::Ref<const Eigen::Matrix<Scalar, Eigen::Dynamic,
                                         Eigen::Dynamic>>& X,
    const Eigen::Matrix<Scalar, 1, Eigen::Dynamic>& mean) {
    return (X.rowwise() - mean).colwise().squaredNorm() /
           static_cast<Scalar>(X.rows());
}

} // namespace internal
} // namespace Skigen

#endif // SKIGEN_CORE_EIGEN_HELPERS_H
