// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors
//
// Internal helper: Platt scaling (sigmoid calibration) via the
// regularised-targets Newton scheme of Lin, Lin & Weng (2007), which is
// what sklearn's _sigmoid_calibration uses.

#ifndef SKIGEN_CALIBRATION_PLATT_SCALING_H
#define SKIGEN_CALIBRATION_PLATT_SCALING_H

#include <Eigen/Core>

#include <cmath>

namespace Skigen {
namespace internal {

/// @brief Fit Platt's sigmoid model `p(y=1|s) = 1 / (1 + exp(A*s + B))`
///   to the training pairs `(scores, targets)` where `targets ∈ {0, 1}`.
///
/// Implements the Newton-with-line-search scheme from Lin, Lin & Weng
/// (2007), "A Note on Platt's Probabilistic Outputs for Support Vector
/// Machines", which is the reference algorithm sklearn uses internally.
///
/// Regularised targets:
/// @f$ t_+ = (n_+ + 1) / (n_+ + 2),\;\; t_- = 1 / (n_- + 2) @f$.
///
/// @param scores 1-D vector of decision-function-style scores.
/// @param targets 1-D vector of binary labels (0 / 1) of the same length.
/// @param A_out On success, set to the fitted slope.
/// @param B_out On success, set to the fitted intercept.
/// @return Number of Newton iterations executed.
template <typename Scalar>
int fit_platt_sigmoid(
    const Eigen::Ref<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>& scores,
    const Eigen::Ref<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>& targets,
    Scalar& A_out, Scalar& B_out)
{
    const Eigen::Index n = scores.size();

    // Count positives / negatives.
    Scalar n_pos = Scalar{0};
    for (Eigen::Index i = 0; i < n; ++i) {
        if (targets(i) > Scalar{0.5}) n_pos += Scalar{1};
    }
    const Scalar n_neg = static_cast<Scalar>(n) - n_pos;

    const Scalar t_hi = (n_pos + Scalar{1}) / (n_pos + Scalar{2});
    const Scalar t_lo = Scalar{1} / (n_neg + Scalar{2});

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> t(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        t(i) = targets(i) > Scalar{0.5} ? t_hi : t_lo;
    }

    // Initial guesses (sklearn convention).
    Scalar A = Scalar{0};
    Scalar B = std::log((n_neg + Scalar{1}) / (n_pos + Scalar{1}));
    constexpr Scalar tol = Scalar{1e-3};
    constexpr int max_iter = 100;
    constexpr Scalar min_step = Scalar{1e-10};
    constexpr Scalar sigma = Scalar{1e-12};

    auto loss_fn = [&](Scalar A_, Scalar B_) {
        Scalar L = Scalar{0};
        for (Eigen::Index i = 0; i < n; ++i) {
            const Scalar fApB = scores(i) * A_ + B_;
            // Numerically-stable form: log(1 + exp(fApB)) and the
            // (t - 1) * fApB component, matching libsvm/sklearn.
            if (fApB >= Scalar{0}) {
                L += t(i) * fApB + std::log(Scalar{1} + std::exp(-fApB));
            } else {
                L += (t(i) - Scalar{1}) * fApB +
                     std::log(Scalar{1} + std::exp(fApB));
            }
        }
        return L;
    };

    Scalar fval = loss_fn(A, B);

    int iter = 0;
    for (; iter < max_iter; ++iter) {
        // Gradient and (positive-definite) Hessian elements.
        Scalar h11 = sigma, h22 = sigma, h21 = Scalar{0};
        Scalar g1 = Scalar{0}, g2 = Scalar{0};
        for (Eigen::Index i = 0; i < n; ++i) {
            const Scalar fApB = scores(i) * A + B;
            Scalar p, q;
            if (fApB >= Scalar{0}) {
                const Scalar e = std::exp(-fApB);
                p = e / (Scalar{1} + e);
                q = Scalar{1} / (Scalar{1} + e);
            } else {
                const Scalar e = std::exp(fApB);
                p = Scalar{1} / (Scalar{1} + e);
                q = e / (Scalar{1} + e);
            }
            const Scalar d2 = p * q;
            h11 += scores(i) * scores(i) * d2;
            h22 += d2;
            h21 += scores(i) * d2;
            const Scalar d1 = t(i) - p;
            g1 += scores(i) * d1;
            g2 += d1;
        }

        if (std::abs(g1) < tol && std::abs(g2) < tol) break;

        // Solve Newton step:  H * d = -g.
        const Scalar det = h11 * h22 - h21 * h21;
        const Scalar dA = -(h22 * g1 - h21 * g2) / det;
        const Scalar dB = -(-h21 * g1 + h11 * g2) / det;
        const Scalar gd = g1 * dA + g2 * dB;

        Scalar stepsize = Scalar{1};
        bool accepted = false;
        while (stepsize >= min_step) {
            const Scalar newA = A + stepsize * dA;
            const Scalar newB = B + stepsize * dB;
            const Scalar newf = loss_fn(newA, newB);
            if (newf < fval + Scalar{0.0001} * stepsize * gd) {
                A = newA; B = newB; fval = newf;
                accepted = true;
                break;
            }
            stepsize /= Scalar{2};
        }
        if (!accepted) break;
    }

    A_out = A;
    B_out = B;
    return iter;
}

/// @brief Apply a fitted Platt sigmoid to a vector of scores, returning the
///   calibrated positive-class probabilities.
template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, 1>
apply_platt_sigmoid(
    const Eigen::Ref<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>& scores,
    Scalar A, Scalar B)
{
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> out(scores.size());
    for (Eigen::Index i = 0; i < scores.size(); ++i) {
        const Scalar fApB = scores(i) * A + B;
        if (fApB >= Scalar{0}) {
            const Scalar e = std::exp(-fApB);
            out(i) = e / (Scalar{1} + e);
        } else {
            const Scalar e = std::exp(fApB);
            out(i) = Scalar{1} / (Scalar{1} + e);
        }
    }
    return out;
}

} // namespace internal
} // namespace Skigen

#endif // SKIGEN_CALIBRATION_PLATT_SCALING_H
