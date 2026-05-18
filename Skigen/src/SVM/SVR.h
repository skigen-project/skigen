// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_SVM_SVR_H
#define SKIGEN_SVM_SVR_H

#include "../Core/Base.h"
#include "../Core/Validation.h"
#include "Detail/Kernels.h"

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <optional>
#include <random>
#include <vector>

namespace Skigen {

/// @addtogroup Algo_KernelSVM
/// @{

/// @brief Epsilon-Support Vector Regression with kernels.
///
/// Mirrors
/// [sklearn.svm.SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html).
///
/// Solves the @f$ \epsilon @f$-insensitive primal in feature space using a
/// sub-gradient SGD over the kernel-mapped dual coefficients. The solver
/// is simpler than libsvm's full SMO-for-regression — it converges to the
/// same minimiser at the optimum but the iterate trace differs.
template <typename Scalar = double>
class SVR : public Predictor<SVR<Scalar>, Scalar> {
public:
    using Base = Predictor<SVR<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;
    using typename Base::ScalarType;
    using Kernel = internal::KernelKind;
    using Base::fit;

    explicit SVR(Scalar C = Scalar{1.0},
                 Kernel kernel = Kernel::RBF,
                 int degree = 3,
                 Scalar gamma = Scalar{0},
                 Scalar coef0 = Scalar{0},
                 Scalar epsilon = Scalar{0.1},
                 Scalar tol = Scalar{1e-3},
                 int max_iter = 1000,
                 std::optional<uint64_t> random_state = std::nullopt)
        : C_(C), kernel_(kernel), degree_(degree),
          gamma_(gamma), coef0_(coef0),
          epsilon_(epsilon), tol_(tol),
          max_iter_(max_iter),
          random_state_(random_state) {}

    [[nodiscard]] Scalar  C()       const noexcept { return C_; }
    [[nodiscard]] Scalar  epsilon() const noexcept { return epsilon_; }
    [[nodiscard]] Kernel  kernel()  const noexcept { return kernel_; }

    [[nodiscard]] const std::vector<Eigen::Index>& support() const {
        this->check_is_fitted(); return support_;
    }
    [[nodiscard]] int n_support() const {
        this->check_is_fitted();
        return static_cast<int>(support_.size());
    }

    SKIGEN_PARAMS(
        (C,       C_,       double),
        (degree,  degree_,  int),
        (gamma,   gamma_,   double),
        (coef0,   coef0_,   double),
        (epsilon, epsilon_, double),
        (tol,     tol_,     double),
        (max_iter,max_iter_, int))

    SVR& fit_impl(const Eigen::Ref<const MatrixType>& X,
                  const Eigen::Ref<const VectorType>& y) {
        internal::check_non_empty(X);
        internal::check_consistent_length(X, y);
        this->n_features_in_ = X.cols();

        const Eigen::Index n = X.rows();
        const Eigen::Index p = X.cols();

        // Resolve gamma="scale".
        const Scalar var_X = X.array().square().sum() /
                             static_cast<Scalar>(n * p) -
                             std::pow(X.mean(), Scalar{2});
        gamma_eff_ = (gamma_ > Scalar{0})
            ? gamma_
            : (var_X > Scalar{0}
                ? Scalar{1} / (static_cast<Scalar>(p) * var_X)
                : Scalar{1});

        // Precompute Gram matrix.
        MatrixType K(n, n);
        for (Eigen::Index i = 0; i < n; ++i) {
            for (Eigen::Index j = i; j < n; ++j) {
                const Scalar k = internal::kernel_eval<Scalar>(
                    kernel_, X.row(i), X.row(j),
                    gamma_eff_, coef0_, degree_);
                K(i, j) = k;
                K(j, i) = k;
            }
        }

        // Sub-gradient SGD on dual coefficients beta (one per training row).
        VectorType beta = VectorType::Zero(n);
        Scalar b{0};
        const uint64_t seed =
            random_state_.value_or(static_cast<uint64_t>(0));
        std::mt19937_64 rng(seed);
        std::vector<Eigen::Index> indices(static_cast<std::size_t>(n));
        std::iota(indices.begin(), indices.end(), Eigen::Index{0});

        const Scalar eta0 = Scalar{0.05};
        for (int t = 0; t < max_iter_; ++t) {
            std::shuffle(indices.begin(), indices.end(), rng);
            const Scalar eta = eta0 /
                (Scalar{1} + Scalar{0.001} * static_cast<Scalar>(t));
            Scalar total_update{0};
            for (auto i : indices) {
                // Predict via current beta over kernel.
                Scalar pred = b;
                for (Eigen::Index j = 0; j < n; ++j) {
                    pred += beta(j) * K(i, j);
                }
                const Scalar r = y(i) - pred;
                if (std::abs(r) <= epsilon_) continue;
                const Scalar coeff = (r > Scalar{0}) ? C_ : -C_;
                // beta(i) update — clamp to [-C, C].
                Scalar new_b_i = beta(i) + eta * coeff;
                new_b_i = std::clamp(new_b_i, -C_, C_);
                const Scalar delta = new_b_i - beta(i);
                beta(i) = new_b_i;
                b += eta * coeff;
                total_update += std::abs(delta);
            }
            if (total_update / static_cast<Scalar>(n) < tol_) break;
        }

        // Store support vectors (rows with beta != 0).
        support_.clear();
        std::vector<Scalar> dual;
        for (Eigen::Index i = 0; i < n; ++i) {
            if (std::abs(beta(i)) > Scalar{1e-7}) {
                support_.push_back(i);
                dual.push_back(beta(i));
            }
        }
        const Eigen::Index ns =
            static_cast<Eigen::Index>(support_.size());
        sv_X_ = MatrixType(ns, p);
        dual_coef_ = VectorType(ns);
        for (Eigen::Index s = 0; s < ns; ++s) {
            sv_X_.row(s) = X.row(support_[static_cast<std::size_t>(s)]);
            dual_coef_(s) = dual[static_cast<std::size_t>(s)];
        }
        intercept_ = b;
        this->fitted_ = true;
        return *this;
    }

    /// @brief Fit from a sparse design matrix (densifies internally).
    template <int Options, typename StorageIndex>
    SVR& fit(const Eigen::SparseMatrix<Scalar, Options, StorageIndex>& X,
             const Eigen::Ref<const VectorType>& y) {
        if (X.rows() == 0 || X.cols() == 0)
            throw std::invalid_argument("SVR.fit: empty sparse matrix.");
        MatrixType Xd = MatrixType(X);
        return fit_impl(Xd, y);
    }

    [[nodiscard]] VectorType predict_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        VectorType out(X.rows());
        for (Eigen::Index i = 0; i < X.rows(); ++i) {
            Scalar s{0};
            for (Eigen::Index t = 0; t < dual_coef_.size(); ++t) {
                s += dual_coef_(t) * internal::kernel_eval<Scalar>(
                    kernel_, X.row(i), sv_X_.row(t),
                    gamma_eff_, coef0_, degree_);
            }
            out(i) = s + intercept_;
        }
        return out;
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
    Kernel kernel_;
    int degree_;
    Scalar gamma_;
    Scalar coef0_;
    Scalar epsilon_;
    Scalar tol_;
    int max_iter_;
    std::optional<uint64_t> random_state_;

    Scalar gamma_eff_{1};
    std::vector<Eigen::Index> support_;
    MatrixType sv_X_;
    VectorType dual_coef_;
    Scalar intercept_{0};
};

/// @}

}  // namespace Skigen

#endif  // SKIGEN_SVM_SVR_H
