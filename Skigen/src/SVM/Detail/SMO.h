// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_SVM_DETAIL_SMO_H
#define SKIGEN_SVM_DETAIL_SMO_H

#include "Kernels.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
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

/// @brief SMO solver for the one-class nu-SVM dual.
///
/// Solves
/// @f[
///   \min_\alpha \tfrac{1}{2}\alpha^\top K \alpha \quad
///   \text{s.t.}\ 0 \le \alpha_i \le \tfrac{1}{\nu n},\ \sum_i \alpha_i = 1.
/// @f]
/// The single sum constraint keeps `alpha_i + alpha_j` invariant under a
/// two-variable update, which is simpler than the signed C-SVM case. The
/// offset `rho` is the average decision value over the free support vectors.
///
/// Inputs:
///   K: precomputed Gram matrix (n × n)
///   nu, tol, max_passes: standard one-class parameters
/// Outputs:
///   alpha: dual coefficients (length n, sum to 1)
///   rho: bias offset
template <typename Scalar>
void smo_one_class(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& K,
                   Scalar nu,
                   Scalar tol,
                   int max_passes,
                   std::optional<uint64_t> random_state,
                   Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& alpha,
                   Scalar& rho) {
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    const Eigen::Index n = K.rows();
    const Scalar upper = Scalar{1} / (nu * static_cast<Scalar>(n));

    // Feasible initialisation: distribute the unit mass evenly across all
    // points (1/n each), which is feasible whenever nu <= 1 (upper >= 1/n).
    // Unlike a fill-the-first-vertices start, the symmetric start lets the
    // optimiser raise outlier alphas to the box bound and lower inlier alphas.
    alpha.setConstant(n, Scalar{1} / static_cast<Scalar>(n));

    auto decision = [&](Eigen::Index i) -> Scalar {
        // g_i = (K alpha)_i
        Scalar g{0};
        for (Eigen::Index j = 0; j < n; ++j) g += alpha(j) * K(i, j);
        return g;
    };

    std::mt19937_64 rng(random_state.value_or(static_cast<uint64_t>(0)));
    std::uniform_int_distribution<Eigen::Index> dist(0, n - 1);

    int passes = 0;
    while (passes < max_passes && n > 1) {
        Eigen::Index num_changed = 0;
        for (Eigen::Index i = 0; i < n; ++i) {
            Eigen::Index j;
            do { j = dist(rng); } while (j == i);

            const Scalar gi = decision(i);
            const Scalar gj = decision(j);
            // With sum(alpha) fixed, alpha_i + alpha_j is invariant; move mass
            // from the higher-gradient vertex to the lower one.
            const Scalar sum_ij = alpha(i) + alpha(j);
            const Scalar eta = K(i, i) + K(j, j) - Scalar{2} * K(i, j);
            if (eta <= Scalar{0}) continue;

            // Unconstrained optimum for alpha_i.
            Scalar new_ai = alpha(i) + (gj - gi) / eta;
            // Box for alpha_i given alpha_j = sum_ij - alpha_i and both in [0, upper].
            const Scalar lo = std::max(Scalar{0}, sum_ij - upper);
            const Scalar hi = std::min(upper, sum_ij);
            if (lo >= hi) continue;
            new_ai = std::clamp(new_ai, lo, hi);
            if (std::abs(new_ai - alpha(i)) < Scalar{1e-7}) continue;
            alpha(i) = new_ai;
            alpha(j) = sum_ij - new_ai;
            ++num_changed;
        }
        if (num_changed == 0) ++passes;
        else passes = 0;
        if (tol <= Scalar{0}) break;
    }

    // Offset rho. The free support vectors (0 < alpha < upper) lie on the
    // margin and define rho. Among the free margin scores choose the one that
    // realises the one-class nu property: at most a nu fraction of all points
    // fall strictly below rho. This keeps rho on the margin while making the
    // decision boundary robust when the auto-scaled kernel flattens the score
    // spread (e.g. extreme outliers inflate the variance-based gamma).
    Vector decisions(n);
    for (Eigen::Index i = 0; i < n; ++i) decisions(i) = decision(i);

    std::vector<Scalar> free_scores;
    for (Eigen::Index i = 0; i < n; ++i) {
        if (alpha(i) > Scalar{1e-8} && alpha(i) < upper - Scalar{1e-8}) {
            free_scores.push_back(decisions(i));
        }
    }

    // Target count of outliers (points strictly below rho).
    const Eigen::Index target_outliers = std::clamp<Eigen::Index>(
        static_cast<Eigen::Index>(std::floor(nu * static_cast<Scalar>(n))),
        Eigen::Index{0}, n - 1);
    Vector sorted = decisions;
    std::sort(sorted.data(), sorted.data() + sorted.size());
    const Scalar quantile_rho = sorted(target_outliers);

    if (free_scores.empty()) {
        rho = quantile_rho;
    } else {
        // Pick the free-margin score closest to the nu-quantile boundary so
        // rho both sits on the margin and respects the nu outlier fraction.
        Scalar best = free_scores.front();
        for (const Scalar s : free_scores) {
            if (std::abs(s - quantile_rho) < std::abs(best - quantile_rho)) {
                best = s;
            }
        }
        rho = std::min(best, quantile_rho);
    }
}

}  // namespace internal
}  // namespace Skigen

#endif  // SKIGEN_SVM_DETAIL_SMO_H
