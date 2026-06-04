// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_SVM_DETAIL_SMO_H
#define SKIGEN_SVM_DETAIL_SMO_H

#include "Kernels.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <random>
#include <vector>

namespace Skigen {
namespace internal {

/// @brief Simplified SMO (Sequential Minimal Optimization) for binary
///   C-SVM classification with kernels.
///
/// This is the simplified "Platt's pseudo-code" SMO from his 1998
/// tutorial — not libsvm's full second-order working-set selection —
/// but the converged solution matches libsvm's at the optimum. Operates
/// on a precomputed kernel cache so per-iteration cost is O(n) inner
/// product lookups.
///
/// Inputs:
///   y: ±1 labels
///   K: precomputed Gram matrix (n × n)
///   C, tol, max_passes: standard SMO parameters
/// Outputs:
///   alpha: dual coefficients (length n)
///   b: bias term
template <typename Scalar>
void smo_binary(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& y,
                const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& K,
                Scalar C,
                Scalar tol,
                int max_passes,
                std::optional<uint64_t> random_state,
                Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& alpha,
                Scalar& b) {
    const Eigen::Index n = K.rows();
    alpha.setZero(n);
    b = Scalar{0};

    auto err_i = [&](Eigen::Index i) -> Scalar {
        // f(x_i) = sum_j alpha_j y_j K(x_i, x_j) + b
        Scalar f{0};
        for (Eigen::Index j = 0; j < n; ++j) {
            f += alpha(j) * y(j) * K(i, j);
        }
        f += b;
        return f - y(i);
    };

    std::mt19937_64 rng(random_state.value_or(
        static_cast<uint64_t>(0)));
    std::uniform_int_distribution<Eigen::Index> dist(0, n - 1);

    int passes = 0;
    while (passes < max_passes) {
        Eigen::Index num_changed = 0;
        for (Eigen::Index i = 0; i < n; ++i) {
            const Scalar Ei = err_i(i);
            const Scalar yi_Ei = y(i) * Ei;
            // KKT violation check.
            if ((yi_Ei < -tol && alpha(i) < C) ||
                (yi_Ei >  tol && alpha(i) > Scalar{0})) {
                // Pick j != i randomly.
                Eigen::Index j;
                do { j = dist(rng); } while (j == i);
                const Scalar Ej = err_i(j);

                const Scalar alpha_i_old = alpha(i);
                const Scalar alpha_j_old = alpha(j);

                Scalar L, H;
                if (y(i) != y(j)) {
                    L = std::max(Scalar{0}, alpha(j) - alpha(i));
                    H = std::min(C, C + alpha(j) - alpha(i));
                } else {
                    L = std::max(Scalar{0}, alpha(j) + alpha(i) - C);
                    H = std::min(C, alpha(j) + alpha(i));
                }
                if (L == H) continue;

                const Scalar eta =
                    Scalar{2} * K(i, j) - K(i, i) - K(j, j);
                if (eta >= Scalar{0}) continue;

                Scalar new_alpha_j =
                    alpha(j) - y(j) * (Ei - Ej) / eta;
                new_alpha_j = std::clamp(new_alpha_j, L, H);
                if (std::abs(new_alpha_j - alpha_j_old) <
                    Scalar{1e-5}) continue;
                alpha(j) = new_alpha_j;
                alpha(i) =
                    alpha_i_old +
                    y(i) * y(j) * (alpha_j_old - new_alpha_j);

                const Scalar b1 =
                    b - Ei -
                    y(i) * (alpha(i) - alpha_i_old) * K(i, i) -
                    y(j) * (alpha(j) - alpha_j_old) * K(i, j);
                const Scalar b2 =
                    b - Ej -
                    y(i) * (alpha(i) - alpha_i_old) * K(i, j) -
                    y(j) * (alpha(j) - alpha_j_old) * K(j, j);
                if (alpha(i) > Scalar{0} && alpha(i) < C) {
                    b = b1;
                } else if (alpha(j) > Scalar{0} && alpha(j) < C) {
                    b = b2;
                } else {
                    b = (b1 + b2) / Scalar{2};
                }
                ++num_changed;
            }
        }
        if (num_changed == 0) ++passes;
        else passes = 0;
    }
}

}  // namespace internal
}  // namespace Skigen

#endif  // SKIGEN_SVM_DETAIL_SMO_H
