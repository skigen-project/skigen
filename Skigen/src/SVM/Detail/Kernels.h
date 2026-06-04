// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_SVM_DETAIL_KERNELS_H
#define SKIGEN_SVM_DETAIL_KERNELS_H

#include <Eigen/Core>

#include <cmath>

namespace Skigen {
namespace internal {

/// @brief Kernel-policy enumeration matching sklearn's `kernel=` strings.
enum class KernelKind { Linear, RBF, Poly, Sigmoid };

/// @brief Evaluate a single kernel value `K(x, z)`.
///
/// `gamma`, `coef0`, `degree` follow sklearn semantics:
/// - Linear:  K(x, z) = x · z
/// - RBF:     K(x, z) = exp(-gamma · ||x - z||²)
/// - Poly:    K(x, z) = (gamma · x · z + coef0)^degree
/// - Sigmoid: K(x, z) = tanh(gamma · x · z + coef0)
template <typename Scalar, typename DerivedX, typename DerivedZ>
Scalar kernel_eval(KernelKind kind,
                   const Eigen::MatrixBase<DerivedX>& x,
                   const Eigen::MatrixBase<DerivedZ>& z,
                   Scalar gamma,
                   Scalar coef0,
                   int degree) {
    switch (kind) {
        case KernelKind::Linear: {
            return x.dot(z);
        }
        case KernelKind::RBF: {
            const Scalar d2 = (x - z).squaredNorm();
            return std::exp(-gamma * d2);
        }
        case KernelKind::Poly: {
            const Scalar t = gamma * x.dot(z) + coef0;
            return std::pow(t, static_cast<Scalar>(degree));
        }
        case KernelKind::Sigmoid: {
            return std::tanh(gamma * x.dot(z) + coef0);
        }
    }
    return Scalar{0};
}

}  // namespace internal
}  // namespace Skigen

#endif  // SKIGEN_SVM_DETAIL_KERNELS_H
