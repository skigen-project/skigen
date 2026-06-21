// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

#ifndef SKIGEN_LINEAR_MODEL_DETAIL_LINEAR_PROGRAM_H
#define SKIGEN_LINEAR_MODEL_DETAIL_LINEAR_PROGRAM_H

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include <algorithm>
#include <cmath>

namespace Skigen {
namespace internal {

/// @brief Result of a standard-form linear-program solve.
template <typename Scalar>
struct LpResult {
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> x;  // primal solution (>= 0)
    int n_iter = 0;
    bool converged = false;
};

/// @brief Solve the standard-form LP
/// @f[
///   \min_{x \ge 0} c^\top x \quad\text{s.t.}\quad A x = b
/// @f]
/// with a primal-dual interior-point method (Mehrotra predictor-corrector).
///
/// `A` is (m × n) with `m < n` and full row rank; `b` is length `m`;
/// `c` is length `n`. The method maintains strictly positive primal `x`
/// and slacks `s` with dual `y`, and follows the central path by Newton
/// steps on the perturbed KKT system. The normal-equations form
/// @f$A D A^\top \Delta y = r@f$ (with @f$D = X S^{-1}@f$) is solved via a
/// Cholesky factorisation each iteration.
///
/// Reference: Mehrotra, "On the Implementation of a Primal-Dual Interior
/// Point Method" (SIAM J. Optim., 1992); Nocedal & Wright, "Numerical
/// Optimization" (2006), Ch. 14, Algorithm 14.3.
///
/// @param A Constraint matrix (m × n).
/// @param b Right-hand side (length m).
/// @param c Cost vector (length n).
/// @param max_iter Maximum interior-point iterations.
/// @param tol Convergence tolerance on the duality measure and residuals.
/// @return Primal solution, iteration count, and convergence flag.
template <typename Scalar>
LpResult<Scalar> solve_standard_lp(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& b,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& c, int max_iter = 200,
    Scalar tol = Scalar{1e-9}) {
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    const Eigen::Index m = A.rows();
    const Eigen::Index n = A.cols();

    LpResult<Scalar> result;
    const Scalar eps = std::numeric_limits<Scalar>::epsilon();

    // Mehrotra's heuristic starting point. Solve the least-squares relaxation
    //   x~ = A^T (A A^T)^{-1} b,   y~ = (A A^T)^{-1} A c,   s~ = c - A^T y~
    // then shift x, s to be strictly positive and balanced.
    Matrix AAt = A * A.transpose();
    for (Eigen::Index i = 0; i < m; ++i)
        AAt(i, i) += Scalar{1e-10} * (Scalar{1} + AAt(i, i));
    Eigen::LLT<Matrix> aat_llt(AAt);
    Vector x = A.transpose() * aat_llt.solve(b);
    Vector y = aat_llt.solve(A * c);
    Vector s = c - A.transpose() * y;

    auto shift_positive = [&](Vector& v) {
        Scalar min_v = v.minCoeff();
        const Scalar delta = std::max(Scalar{-1.5} * min_v, Scalar{0});
        v.array() += delta;
    };
    shift_positive(x);
    shift_positive(s);
    // Final centering shift so the pairwise products are balanced.
    {
        const Scalar xs = x.dot(s);
        const Scalar sum_s = s.sum();
        const Scalar sum_x = x.sum();
        const Scalar dx = (sum_s > eps) ? Scalar{0.5} * xs / sum_s : Scalar{1};
        const Scalar ds = (sum_x > eps) ? Scalar{0.5} * xs / sum_x : Scalar{1};
        x.array() += std::max(dx, Scalar{1});
        s.array() += std::max(ds, Scalar{1});
    }

    for (int iter = 0; iter < max_iter; ++iter) {
        // Residuals.
        const Vector rb = A * x - b;            // primal (m)
        const Vector rc = A.transpose() * y + s - c;  // dual (n)
        const Scalar mu = x.dot(s) / static_cast<Scalar>(n);

        // Convergence test.
        const Scalar rb_norm = rb.norm() / (Scalar{1} + b.norm());
        const Scalar rc_norm = rc.norm() / (Scalar{1} + c.norm());
        if (rb_norm < tol && rc_norm < tol && mu < tol) {
            result.converged = true;
            result.n_iter = iter;
            break;
        }

        // Diagonal D = X S^{-1}; normal matrix A D A^T.
        Vector d(n);
        for (Eigen::Index i = 0; i < n; ++i)
            d(i) = x(i) / std::max(s(i), eps);

        Matrix ADAt = A * d.asDiagonal() * A.transpose();
        // Adaptive Tikhonov regularisation keeps the normal matrix SPD even
        // when variables approach their bounds (degenerate LP vertices).
        const Scalar reg = std::max(mu, eps) * Scalar{1e-6} +
                           Scalar{1e-12};
        for (Eigen::Index i = 0; i < m; ++i)
            ADAt(i, i) += reg * (Scalar{1} + ADAt(i, i));
        Eigen::LLT<Matrix> llt(ADAt);
        if (llt.info() != Eigen::Success) {
            // Fall back to a stronger ridge if Cholesky failed.
            for (Eigen::Index i = 0; i < m; ++i)
                ADAt(i, i) += Scalar{1e-6} * (Scalar{1} + ADAt(i, i));
            llt.compute(ADAt);
        }

        // --- Newton step solver (Nocedal & Wright, eq. 14.30) ---
        // Eliminating ds = -rc - A^T dy and dx = -S^{-1}(rxs + X ds):
        //   (A D A^T) dy = -rb + A S^{-1} rxs - A D rc,   D = X S^{-1}.
        auto solve_step = [&](const Vector& rxs) {
            Vector s_inv_rxs(n), d_rc(n);
            for (Eigen::Index i = 0; i < n; ++i) {
                const Scalar si = std::max(s(i), eps);
                s_inv_rxs(i) = rxs(i) / si;
                d_rc(i) = d(i) * rc(i);
            }
            Vector rhs = -rb + A * s_inv_rxs - A * d_rc;
            Vector dy = llt.solve(rhs);
            Vector ds = -rc - A.transpose() * dy;
            Vector dx(n);
            for (Eigen::Index i = 0; i < n; ++i)
                dx(i) = -(rxs(i) + x(i) * ds(i)) / std::max(s(i), eps);
            return std::make_tuple(dx, dy, ds);
        };

        Vector rxs_aff = x.cwiseProduct(s);  // X S e
        auto [dx_a, dy_a, ds_a] = solve_step(rxs_aff);

        // Affine step lengths.
        auto step_len = [&](const Vector& v, const Vector& dv) {
            Scalar a = Scalar{1};
            for (Eigen::Index i = 0; i < v.size(); ++i)
                if (dv(i) < Scalar{0})
                    a = std::min(a, -v(i) / dv(i));
            return a;
        };
        const Scalar ap_aff = step_len(x, dx_a);
        const Scalar ad_aff = step_len(s, ds_a);

        // Centring parameter sigma from the affine duality gap.
        Scalar mu_aff{0};
        for (Eigen::Index i = 0; i < n; ++i)
            mu_aff += (x(i) + ap_aff * dx_a(i)) * (s(i) + ad_aff * ds_a(i));
        mu_aff /= static_cast<Scalar>(n);
        const Scalar sigma = (mu > eps)
            ? std::pow(mu_aff / mu, Scalar{3}) : Scalar{0};

        // --- Corrector step ---
        // rxs = X S e + dX dS e - sigma mu e
        Vector rxs(n);
        for (Eigen::Index i = 0; i < n; ++i)
            rxs(i) = x(i) * s(i) + dx_a(i) * ds_a(i) - sigma * mu;
        auto [dx, dy, ds] = solve_step(rxs);

        const Scalar ap = Scalar{0.99} * step_len(x, dx);
        const Scalar ad = Scalar{0.99} * step_len(s, ds);

        x += ap * dx;
        y += ad * dy;
        s += ad * ds;

        // Keep primal/slack strictly interior to avoid division blow-ups at
        // degenerate vertices.
        const Scalar floor = eps * (Scalar{1} + x.cwiseAbs().maxCoeff());
        for (Eigen::Index i = 0; i < n; ++i) {
            if (x(i) < floor) x(i) = floor;
            if (s(i) < floor) s(i) = floor;
        }
        result.n_iter = iter + 1;
    }

    result.x = x;
    return result;
}

}  // namespace internal
}  // namespace Skigen

#endif  // SKIGEN_LINEAR_MODEL_DETAIL_LINEAR_PROGRAM_H
