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

/// @brief libsvm-style SMO for binary C-SVM with second-order working-set
///   selection (WSS3).
///
/// Solves the C-SVM dual
/// @f[
///   \min_\alpha \tfrac12 \alpha^\top Q \alpha - e^\top \alpha,\quad
///   0 \le \alpha_i \le C,\ y^\top \alpha = 0,\quad Q_{ij}=y_i y_j K_{ij}.
/// @f]
/// Each iteration selects the maximal-violating index `i` from the
/// up-set and the index `j` that minimises the second-order objective
/// decrease (Fan, Chen & Lin, "Working Set Selection Using Second Order
/// Information for Training SVM", JMLR 2005 — the selection libsvm and
/// scikit-learn use). The projected gradient gap drives the stopping rule,
/// so the solver terminates at the true KKT point rather than after a fixed
/// number of passes. The intercept follows libsvm's free-SV averaging.
///
/// Inputs:
///   y: ±1 labels
///   K: precomputed Gram matrix (n × n)
///   C, tol: regularisation and KKT tolerance
///   max_iter: hard cap on iterations (libsvm uses max(1e7, 100*n))
/// Outputs:
///   alpha: dual coefficients (length n)
///   b: bias term
template <typename Scalar>
void smo_binary_wss3(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& y,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& K, Scalar C,
    Scalar tol, int max_iter,
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& alpha, Scalar& b) {
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    const Eigen::Index n = K.rows();
    alpha.setZero(n);

    // Gradient G_i = (Q alpha)_i - 1; starts at -1 since alpha = 0.
    Vector G = Vector::Constant(n, Scalar{-1});

    const Scalar tau = Scalar{1e-12};  // libsvm's TAU guard for eta.
    const Eigen::Index iter_cap =
        (max_iter > 0) ? static_cast<Eigen::Index>(max_iter)
                       : std::max<Eigen::Index>(10000000, 100 * n);

    auto is_up = [&](Eigen::Index t) {
        return (y(t) > Scalar{0}) ? (alpha(t) < C) : (alpha(t) > Scalar{0});
    };
    auto is_low = [&](Eigen::Index t) {
        return (y(t) > Scalar{0}) ? (alpha(t) > Scalar{0}) : (alpha(t) < C);
    };

    for (Eigen::Index iter = 0; iter < iter_cap; ++iter) {
        // --- Working-set selection (WSS3) ---
        // i = argmax over up-set of -y_t G_t.
        Eigen::Index i = -1;
        Scalar Gmax = -std::numeric_limits<Scalar>::infinity();
        Scalar Gmin = std::numeric_limits<Scalar>::infinity();
        for (Eigen::Index t = 0; t < n; ++t) {
            if (is_up(t)) {
                const Scalar v = -y(t) * G(t);
                if (v >= Gmax) { Gmax = v; i = t; }
            }
        }
        // j minimises the second-order objective decrease over the low-set.
        Eigen::Index j = -1;
        Scalar obj_min = std::numeric_limits<Scalar>::infinity();
        for (Eigen::Index t = 0; t < n; ++t) {
            if (is_low(t)) {
                const Scalar grad_diff = Gmax + y(t) * G(t);
                if (-y(t) * G(t) <= Gmin) Gmin = -y(t) * G(t);
                if (grad_diff > Scalar{0} && i >= 0) {
                    Scalar quad =
                        K(i, i) + K(t, t) - Scalar{2} * K(i, t);
                    if (quad <= Scalar{0}) quad = tau;
                    const Scalar obj = -(grad_diff * grad_diff) / quad;
                    if (obj <= obj_min) { obj_min = obj; j = t; }
                }
            }
        }

        // Stopping: projected-gradient gap below tolerance.
        if (Gmax - Gmin < tol || i < 0 || j < 0) break;

        // --- Two-variable update on (i, j) ---
        Scalar quad = K(i, i) + K(j, j) - Scalar{2} * K(i, j);
        if (quad <= Scalar{0}) quad = tau;
        const Scalar old_ai = alpha(i);
        const Scalar old_aj = alpha(j);

        if (y(i) != y(j)) {
            const Scalar delta = (-G(i) - G(j)) / quad;
            const Scalar diff = alpha(i) - alpha(j);
            alpha(i) += delta;
            alpha(j) += delta;
            if (diff > Scalar{0}) {
                if (alpha(j) < Scalar{0}) { alpha(j) = Scalar{0}; alpha(i) = diff; }
            } else {
                if (alpha(i) < Scalar{0}) { alpha(i) = Scalar{0}; alpha(j) = -diff; }
            }
            if (diff > Scalar{0}) {
                if (alpha(i) > C) { alpha(i) = C; alpha(j) = C - diff; }
            } else {
                if (alpha(j) > C) { alpha(j) = C; alpha(i) = C + diff; }
            }
        } else {
            const Scalar delta = (G(i) - G(j)) / quad;
            const Scalar sum = alpha(i) + alpha(j);
            alpha(i) -= delta;
            alpha(j) += delta;
            if (sum > C) {
                if (alpha(i) > C) { alpha(i) = C; alpha(j) = sum - C; }
            } else {
                if (alpha(j) < Scalar{0}) { alpha(j) = Scalar{0}; alpha(i) = sum; }
            }
            if (sum > C) {
                if (alpha(j) > C) { alpha(j) = C; alpha(i) = sum - C; }
            } else {
                if (alpha(i) < Scalar{0}) { alpha(i) = Scalar{0}; alpha(j) = sum; }
            }
        }

        // --- Gradient update ---
        const Scalar dai = alpha(i) - old_ai;
        const Scalar daj = alpha(j) - old_aj;
        for (Eigen::Index t = 0; t < n; ++t) {
            G(t) += y(t) * y(i) * K(t, i) * dai +
                    y(t) * y(j) * K(t, j) * daj;
        }
    }

    // --- Intercept: average the free-SV margins (libsvm's rho). ---
    Scalar ub = std::numeric_limits<Scalar>::infinity();
    Scalar lb = -std::numeric_limits<Scalar>::infinity();
    Scalar sum_free{0};
    Eigen::Index n_free = 0;
    for (Eigen::Index t = 0; t < n; ++t) {
        const Scalar yG = y(t) * G(t);
        if (alpha(t) >= C) {
            if (y(t) > Scalar{0}) lb = std::max(lb, yG);
            else ub = std::min(ub, yG);
        } else if (alpha(t) <= Scalar{0}) {
            if (y(t) < Scalar{0}) lb = std::max(lb, yG);
            else ub = std::min(ub, yG);
        } else {
            ++n_free;
            sum_free += yG;
        }
    }
    const Scalar rho = (n_free > 0)
        ? sum_free / static_cast<Scalar>(n_free)
        : (ub + lb) / Scalar{2};
    // libsvm stores -rho as the bias for the decision sum_i alpha_i y_i K + b.
    b = -rho;
}

/// @brief Simplified SMO (Sequential Minimal Optimization) for binary
///   C-SVM classification with kernels.
///
/// Delegates to the libsvm-style second-order working-set selection solver
/// (`smo_binary_wss3`). The legacy `max_passes` parameter is mapped to an
/// iteration cap; `random_state` is unused by the deterministic WSS3 path
/// and retained only for API compatibility.
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
    (void)random_state;
    // Map the legacy pass budget to an iteration cap (>= libsvm's floor).
    const int max_iter = (max_passes > 0)
        ? std::max(max_passes, 1000) * static_cast<int>(K.rows() + 1)
        : 0;
    smo_binary_wss3<Scalar>(y, K, C, tol > Scalar{0} ? tol : Scalar{1e-3},
                            max_iter, alpha, b);
}

/// @brief Legacy simplified Platt SMO (kept for reference / fallback).
template <typename Scalar>
void smo_binary_platt(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& y,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& K, Scalar C,
    Scalar tol, int max_passes, std::optional<uint64_t> random_state,
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& alpha, Scalar& b) {
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

/// @brief SMO solver for the nu-SVM binary classification dual.
///
/// Solves the nu-classification dual
/// @f[
///   \min_\alpha \tfrac{1}{2}\alpha^\top (y y^\top \odot K) \alpha \quad
///   \text{s.t.}\ 0 \le \alpha_i \le 1,\
///   \sum_i \alpha_i y_i = 0,\ \sum_i \alpha_i = \nu n.
/// @f]
/// Working on @f$ \beta_i = \alpha_i y_i @f$ turns the two equality
/// constraints into @f$ \sum_{y_i=+1}\alpha_i = \sum_{y_i=-1}\alpha_i =
/// \nu n / 2 @f$, so two-variable updates must keep the per-class mass fixed:
/// pairs are selected within the same class. The decision threshold `b` and
/// margin `rho` come from the free support vectors of each class.
///
/// Inputs:
///   y: ±1 labels
///   K: precomputed Gram matrix (n × n)
///   nu, tol, max_passes: standard nu-SVM parameters
/// Outputs:
///   alpha: dual coefficients in [0, 1] (length n)
///   b: bias term (so decision = sum_i alpha_i y_i K(x,x_i) + b)
template <typename Scalar>
bool smo_nu_classification(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& y,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& K,
    Scalar nu,
    Scalar tol,
    int max_passes,
    std::optional<uint64_t> random_state,
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& alpha,
    Scalar& b) {
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    const Eigen::Index n = K.rows();

    // Count classes and the feasible per-class mass nu*n/2.
    Eigen::Index n_pos = 0, n_neg = 0;
    for (Eigen::Index i = 0; i < n; ++i) (y(i) > Scalar{0} ? n_pos : n_neg)++;
    if (n_pos == 0 || n_neg == 0) return false;

    const Scalar mass = nu * static_cast<Scalar>(n) / Scalar{2};
    // Feasibility: mass cannot exceed the count of either class (box [0,1]).
    if (mass > static_cast<Scalar>(std::min(n_pos, n_neg))) {
        return false;  // infeasible nu for this class balance
    }

    // Feasible init: spread the per-class mass uniformly inside each class.
    alpha.setZero(n);
    const Scalar a_pos = mass / static_cast<Scalar>(n_pos);
    const Scalar a_neg = mass / static_cast<Scalar>(n_neg);
    for (Eigen::Index i = 0; i < n; ++i) {
        alpha(i) = (y(i) > Scalar{0}) ? a_pos : a_neg;
    }

    auto grad = [&](Eigen::Index i) -> Scalar {
        // d/d alpha_i of 0.5 a^T (yy^T ⊙ K) a = y_i sum_j alpha_j y_j K(i,j)
        Scalar g{0};
        for (Eigen::Index j = 0; j < n; ++j) g += alpha(j) * y(j) * K(i, j);
        return y(i) * g;
    };

    std::mt19937_64 rng(random_state.value_or(static_cast<uint64_t>(0)));
    std::uniform_int_distribution<Eigen::Index> dist(0, n - 1);

    int passes = 0;
    while (passes < max_passes && n > 1) {
        Eigen::Index num_changed = 0;
        for (Eigen::Index i = 0; i < n; ++i) {
            // Select j in the same class as i to preserve per-class mass.
            Eigen::Index j;
            int tries = 0;
            do { j = dist(rng); ++tries; }
            while ((j == i || y(j) != y(i)) && tries < 4 * static_cast<int>(n));
            if (j == i || y(j) != y(i)) continue;

            const Scalar gi = grad(i);
            const Scalar gj = grad(j);
            const Scalar sum_ij = alpha(i) + alpha(j);  // invariant
            const Scalar eta = K(i, i) + K(j, j) - Scalar{2} * K(i, j);
            if (eta <= Scalar{0}) continue;

            Scalar new_ai = alpha(i) + (gj - gi) / eta;
            const Scalar lo = std::max(Scalar{0}, sum_ij - Scalar{1});
            const Scalar hi = std::min(Scalar{1}, sum_ij);
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

    // Recover b and rho from the free support vectors. For nu-SVM:
    //   decision_i = sum_j alpha_j y_j K(i,j)
    //   on the +1 free margin: decision = rho - b
    //   on the -1 free margin: decision = -rho - b
    // so b = -(d_pos + d_neg)/2 where d_pos/d_neg are mean free decisions.
    Vector decision(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        Scalar d{0};
        for (Eigen::Index j = 0; j < n; ++j) d += alpha(j) * y(j) * K(i, j);
        decision(i) = d;
    }
    Scalar sum_pos{0}, sum_neg{0};
    Eigen::Index cp = 0, cn = 0;
    for (Eigen::Index i = 0; i < n; ++i) {
        const bool free = alpha(i) > Scalar{1e-6} &&
                          alpha(i) < Scalar{1} - Scalar{1e-6};
        if (!free) continue;
        if (y(i) > Scalar{0}) { sum_pos += decision(i); ++cp; }
        else { sum_neg += decision(i); ++cn; }
    }
    if (cp > 0 && cn > 0) {
        const Scalar d_pos = sum_pos / static_cast<Scalar>(cp);
        const Scalar d_neg = sum_neg / static_cast<Scalar>(cn);
        b = -(d_pos + d_neg) / Scalar{2};
    } else {
        // Fallback: threshold so the decision separates the class means.
        Scalar mp{0}, mn{0};
        Eigen::Index np2 = 0, nn2 = 0;
        for (Eigen::Index i = 0; i < n; ++i) {
            if (y(i) > Scalar{0}) { mp += decision(i); ++np2; }
            else { mn += decision(i); ++nn2; }
        }
        if (np2 > 0) mp /= static_cast<Scalar>(np2);
        if (nn2 > 0) mn /= static_cast<Scalar>(nn2);
        b = -(mp + mn) / Scalar{2};
    }
    return true;
}

}  // namespace internal
}  // namespace Skigen

#endif  // SKIGEN_SVM_DETAIL_SMO_H
