// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_SVM_LINEAR_SVR_H
#define SKIGEN_SVM_LINEAR_SVR_H

#include "../Core/Base.h"
#include "../Core/Validation.h"

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <optional>
#include <random>
#include <vector>

namespace Skigen {

/// @addtogroup Algo_LinearSVM
/// @{

/// @brief Linear epsilon-insensitive Support Vector Regression.
///
/// Mirrors
/// [sklearn.svm.LinearSVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html).
///
/// Solves
/// @f$ \min_w \tfrac{1}{2}\|w\|^2 + C \sum_i L(y_i - w \cdot x_i - b) @f$
/// where @f$ L @f$ is the @f$ \epsilon @f$-insensitive loss
/// (`loss="epsilon_insensitive"`, the sklearn default) — the residual
/// contributes only when @f$ |r| > \epsilon @f$.
template <typename Scalar = double>
class LinearSVR : public Predictor<LinearSVR<Scalar>, Scalar> {
public:
    using Base = Predictor<LinearSVR<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;
    using typename Base::ScalarType;
    using Base::fit;

    enum class Loss { EpsilonInsensitive, SquaredEpsilonInsensitive };

    explicit LinearSVR(Scalar C = Scalar{1.0},
                       Scalar epsilon = Scalar{0},
                       Loss loss = Loss::EpsilonInsensitive,
                       Scalar tol = Scalar{1e-4},
                       int max_iter = 1000,
                       bool fit_intercept = true,
                       std::optional<uint64_t> random_state = std::nullopt)
        : C_(C), epsilon_(epsilon), loss_(loss), tol_(tol),
          max_iter_(max_iter),
          fit_intercept_(fit_intercept),
          random_state_(random_state) {}

    [[nodiscard]] Scalar C() const noexcept { return C_; }
    [[nodiscard]] Scalar epsilon() const noexcept { return epsilon_; }

    [[nodiscard]] const RowVectorType& coef() const {
        this->check_is_fitted(); return coef_;
    }
    [[nodiscard]] Scalar intercept() const {
        this->check_is_fitted(); return intercept_;
    }

    SKIGEN_PARAMS(
        (C,             C_,             double),
        (epsilon,       epsilon_,       double),
        (tol,           tol_,           double),
        (max_iter,      max_iter_,      int),
        (fit_intercept, fit_intercept_, bool))

    LinearSVR& fit_impl(const Eigen::Ref<const MatrixType>& X,
                        const Eigen::Ref<const VectorType>& y) {
        internal::check_non_empty(X);
        internal::check_consistent_length(X, y);
        this->n_features_in_ = X.cols();

        const Eigen::Index n = X.rows();
        const Eigen::Index p = X.cols();
        coef_      = RowVectorType::Zero(p);
        intercept_ = Scalar{0};

        const uint64_t seed =
            random_state_.value_or(static_cast<uint64_t>(42));
        std::mt19937_64 rng(seed);
        std::vector<Eigen::Index> indices(static_cast<std::size_t>(n));
        std::iota(indices.begin(), indices.end(), Eigen::Index{0});

        // Per-row gradient steps for the eps-insensitive losses are
        // bounded by C * ||x_i|| (linear hinge) or 2C * (|r| - eps) * ||x_i||
        // (squared hinge). Scaling the learning rate by 1/(C * max ||x_i||²)
        // keeps the iterates stable and avoids the divergence that a fixed
        // eta produces when C * max ||x||² is large.
        Scalar max_norm_sq = Scalar{1};
        for (Eigen::Index i = 0; i < n; ++i) {
            const Scalar s = X.row(i).squaredNorm() +
                             (fit_intercept_ ? Scalar{1} : Scalar{0});
            if (s > max_norm_sq) max_norm_sq = s;
        }
        const Scalar lipschitz_bound =
            (loss_ == Loss::SquaredEpsilonInsensitive)
                ? Scalar{2} * C_ * max_norm_sq
                : C_ * max_norm_sq;
        const Scalar eta0 = Scalar{1} / (lipschitz_bound + Scalar{1e-12});
        const Scalar reg  = Scalar{1} /
                            (C_ * static_cast<Scalar>(n));

        for (int t = 0; t < max_iter_; ++t) {
            std::shuffle(indices.begin(), indices.end(), rng);
            const Scalar eta = eta0 /
                (Scalar{1} + reg * eta0 * static_cast<Scalar>(t));
            Scalar total_update{0};
            for (auto idx : indices) {
                const Scalar pred =
                    (X.row(idx) * coef_.transpose())(0) + intercept_;
                const Scalar r = y(idx) - pred;
                const Scalar absr = std::abs(r);
                // Inside the eps-band: only the L2 shrinkage is applied.
                if (absr <= epsilon_) {
                    coef_ *= (Scalar{1} - eta * reg);
                    continue;
                }
                Scalar coeff{0};
                if (loss_ == Loss::EpsilonInsensitive) {
                    coeff = (r > Scalar{0}) ? C_ : -C_;
                } else {
                    coeff = Scalar{2} * C_ * (absr - epsilon_) *
                            ((r > Scalar{0}) ? Scalar{1} : Scalar{-1});
                }
                coef_ += eta * (coeff * X.row(idx) - reg * coef_);
                if (fit_intercept_) intercept_ += eta * coeff;
                total_update += std::abs(eta * coeff);
            }
            (void)p;
            if (total_update / static_cast<Scalar>(n) < tol_) break;
        }

        this->fitted_ = true;
        return *this;
    }

    /// @brief Fit from a sparse design matrix (densifies internally).
    template <int Options, typename StorageIndex>
    LinearSVR& fit(const Eigen::SparseMatrix<Scalar, Options, StorageIndex>& X,
                   const Eigen::Ref<const VectorType>& y) {
        if (X.rows() == 0 || X.cols() == 0)
            throw std::invalid_argument("LinearSVR.fit: empty sparse matrix.");
        MatrixType Xd = MatrixType(X);
        return fit_impl(Xd, y);
    }

    [[nodiscard]] VectorType predict_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        return (X * coef_.transpose()).array() + intercept_;
    }

    [[nodiscard]] Scalar score_impl(
        const Eigen::Ref<const MatrixType>& X,
        const Eigen::Ref<const VectorType>& y) const {
        VectorType yh = predict_impl(X);
        const Scalar ym = y.mean();
        const Scalar ss_res = (y - yh).squaredNorm();
        const Scalar ss_tot =
            (y.array() - ym).matrix().squaredNorm();
        if (ss_tot == Scalar{0}) return Scalar{0};
        return Scalar{1} - ss_res / ss_tot;
    }

private:
    Scalar C_;
    Scalar epsilon_;
    Loss loss_;
    Scalar tol_;
    int max_iter_;
    bool fit_intercept_;
    std::optional<uint64_t> random_state_;

    RowVectorType coef_;
    Scalar intercept_{0};
};

/// @}

}  // namespace Skigen

#endif  // SKIGEN_SVM_LINEAR_SVR_H
