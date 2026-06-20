// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

#ifndef SKIGEN_OPTIMIZATION_NELDER_MEAD_H
#define SKIGEN_OPTIMIZATION_NELDER_MEAD_H

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace Skigen {

/// @defgroup Algo_NelderMead Nelder-Mead Optimization
/// @ingroup Optimization
/// @brief Derivative-free simplex optimization.
/// @{

/// @brief Derivative-free Nelder-Mead simplex minimizer.
///
/// Implements the classical reflection, expansion, contraction, and shrink
/// steps from Nelder and Mead (1965). The implementation is header-only,
/// deterministic for deterministic objectives, and accepts arbitrary Eigen
/// dynamic vectors.
///
/// ### Examples
///
/// @snippet nelder_mead.cpp example_nelder_mead
template <typename Scalar = double>
class NelderMead {
public:
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    struct Result {
        VectorType x;
        Scalar fun = std::numeric_limits<Scalar>::quiet_NaN();
        int nit = 0;
        int nfev = 0;
        bool success = false;
        std::string message;
        MatrixType final_simplex;
        VectorType final_values;
    };

    explicit NelderMead(int max_iter = 1000,
                        Scalar xatol = Scalar{1e-8},
                        Scalar fatol = Scalar{1e-8},
                        Scalar initial_step = Scalar{0.05})
        : max_iter_(max_iter),
          xatol_(xatol),
          fatol_(fatol),
          initial_step_(initial_step) {
        if (max_iter_ <= 0) {
            throw std::invalid_argument("NelderMead: max_iter must be positive.");
        }
        if (xatol_ < Scalar{0} || fatol_ < Scalar{0} || initial_step_ <= Scalar{0}) {
            throw std::invalid_argument(
                "NelderMead: tolerances must be non-negative and initial_step positive.");
        }
    }

    [[nodiscard]] int max_iter() const noexcept { return max_iter_; }
    [[nodiscard]] Scalar xatol() const noexcept { return xatol_; }
    [[nodiscard]] Scalar fatol() const noexcept { return fatol_; }
    [[nodiscard]] Scalar initial_step() const noexcept { return initial_step_; }

    template <typename Objective>
    [[nodiscard]] Result minimize(const Eigen::Ref<const VectorType>& x0,
                                  Objective&& objective) const {
        if (x0.size() == 0) {
            throw std::invalid_argument("NelderMead.minimize: x0 must be non-empty.");
        }
        if (!x0.allFinite()) {
            throw std::invalid_argument("NelderMead.minimize: x0 must be finite.");
        }
        MatrixType simplex(x0.size() + 1, x0.size());
        simplex.row(0) = x0.transpose();
        for (Eigen::Index dim = 0; dim < x0.size(); ++dim) {
            VectorType vertex = x0;
            const Scalar scale = std::max(std::abs(x0(dim)), Scalar{1});
            vertex(dim) += initial_step_ * scale;
            simplex.row(dim + 1) = vertex.transpose();
        }
        return minimize_simplex(simplex, std::forward<Objective>(objective));
    }

    template <typename Objective>
    [[nodiscard]] Result minimize_simplex(MatrixType simplex,
                                          Objective&& objective) const {
        const Eigen::Index vertices = simplex.rows();
        const Eigen::Index dimension = simplex.cols();
        if (dimension == 0 || vertices != dimension + 1) {
            throw std::invalid_argument(
                "NelderMead.minimize_simplex: simplex must have shape (n + 1, n).");
        }
        if (!simplex.allFinite()) {
            throw std::invalid_argument("NelderMead.minimize_simplex: simplex must be finite.");
        }

        VectorType values(vertices);
        int evaluations = 0;
        for (Eigen::Index i = 0; i < vertices; ++i) {
            values(i) = objective(simplex.row(i).transpose());
            ++evaluations;
            if (!std::isfinite(values(i))) {
                throw std::invalid_argument("NelderMead: objective returned non-finite value.");
            }
        }

        Result result;
        result.success = false;
        result.message = "maximum iterations reached";

        constexpr Scalar alpha = Scalar{1};
        constexpr Scalar gamma = Scalar{2};
        constexpr Scalar rho = Scalar{0.5};
        constexpr Scalar sigma = Scalar{0.5};

        for (int iteration = 0; iteration < max_iter_; ++iteration) {
            const auto [best, worst, second_worst] = rank_vertices(values);
            const Scalar f_best = values(best);
            const Scalar f_worst = values(worst);
            const Scalar x_spread = simplex_spread(simplex, best);
            const Scalar f_spread = (values.array() - f_best).abs().maxCoeff();
            if (x_spread <= xatol_ && f_spread <= fatol_) {
                result.success = true;
                result.message = "converged";
                result.nit = iteration;
                break;
            }

            VectorType centroid = VectorType::Zero(dimension);
            for (Eigen::Index i = 0; i < vertices; ++i) {
                if (i != worst) centroid += simplex.row(i).transpose();
            }
            centroid /= static_cast<Scalar>(dimension);

            const VectorType worst_vertex = simplex.row(worst).transpose();
            const VectorType reflected = centroid + alpha * (centroid - worst_vertex);
            const Scalar f_reflected = objective(reflected);
            ++evaluations;

            if (f_reflected < f_best) {
                const VectorType expanded = centroid + gamma * (reflected - centroid);
                const Scalar f_expanded = objective(expanded);
                ++evaluations;
                if (f_expanded < f_reflected) {
                    simplex.row(worst) = expanded.transpose();
                    values(worst) = f_expanded;
                } else {
                    simplex.row(worst) = reflected.transpose();
                    values(worst) = f_reflected;
                }
            } else if (f_reflected < values(second_worst)) {
                simplex.row(worst) = reflected.transpose();
                values(worst) = f_reflected;
            } else {
                const bool outside = f_reflected < f_worst;
                VectorType contracted(dimension);
                if (outside) {
                    contracted = (centroid + rho * (reflected - centroid)).eval();
                } else {
                    contracted = (centroid - rho * (centroid - worst_vertex)).eval();
                }
                const Scalar f_contracted = objective(contracted);
                ++evaluations;
                if (f_contracted < (outside ? f_reflected : f_worst)) {
                    simplex.row(worst) = contracted.transpose();
                    values(worst) = f_contracted;
                } else {
                    const VectorType best_vertex = simplex.row(best).transpose();
                    for (Eigen::Index i = 0; i < vertices; ++i) {
                        if (i == best) continue;
                        VectorType shrunk = best_vertex + sigma * (simplex.row(i).transpose() - best_vertex);
                        simplex.row(i) = shrunk.transpose();
                        values(i) = objective(shrunk);
                        ++evaluations;
                    }
                }
            }

            result.nit = iteration + 1;
        }

        const auto [best, _, __] = rank_vertices(values);
        (void)_; (void)__;
        result.x = simplex.row(best).transpose();
        result.fun = values(best);
        result.nfev = evaluations;
        result.final_simplex = simplex;
        result.final_values = values;
        if (!result.success && result.nit < max_iter_) {
            result.success = true;
            result.message = "converged";
        }
        return result;
    }

private:
    [[nodiscard]] static std::tuple<Eigen::Index, Eigen::Index, Eigen::Index>
    rank_vertices(const VectorType& values) {
        Eigen::Index best = 0;
        Eigen::Index worst = 0;
        for (Eigen::Index i = 1; i < values.size(); ++i) {
            if (values(i) < values(best)) best = i;
            if (values(i) > values(worst)) worst = i;
        }
        Eigen::Index second_worst = best == 0 ? 1 : 0;
        for (Eigen::Index i = 0; i < values.size(); ++i) {
            if (i == worst) continue;
            if (second_worst == worst || values(i) > values(second_worst)) second_worst = i;
        }
        return {best, worst, second_worst};
    }

    [[nodiscard]] static Scalar simplex_spread(const MatrixType& simplex,
                                               Eigen::Index best) {
        Scalar spread = Scalar{0};
        for (Eigen::Index i = 0; i < simplex.rows(); ++i) {
            spread = std::max(spread,
                              (simplex.row(i) - simplex.row(best)).cwiseAbs().maxCoeff());
        }
        return spread;
    }

    int max_iter_;
    Scalar xatol_;
    Scalar fatol_;
    Scalar initial_step_;
};

/// @}

}  // namespace Skigen

#endif  // SKIGEN_OPTIMIZATION_NELDER_MEAD_H
