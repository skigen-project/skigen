// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

#ifndef SKIGEN_SVM_NU_SVR_H
#define SKIGEN_SVM_NU_SVR_H

#include "../Core/Base.h"
#include "../Core/Validation.h"
#include "Detail/Kernels.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

namespace Skigen {

/// @addtogroup Algo_KernelSVM
/// @{

/// @brief Nu-Support Vector Regression with kernels.
///
/// In nu-SVR, `nu` controls the fraction of support vectors and bounds the
/// fraction of points outside the epsilon tube; the tube width epsilon is
/// learned from the data rather than fixed. Skigen fits the dual coefficients
/// with the same sub-gradient solver as `SVR`, choosing epsilon as the
/// `(1 - nu)` quantile of the absolute residuals so that roughly a `nu`
/// fraction of training points lie outside the tube.
///
/// Mirrors the dense core of
/// [sklearn.svm.NuSVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVR.html).
template <typename Scalar = double>
class NuSVR : public Predictor<NuSVR<Scalar>, Scalar> {
public:
    using Base = Predictor<NuSVR<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::IndexType;
    using typename Base::ScalarType;
    using Kernel = internal::KernelKind;
    using Base::fit;

    explicit NuSVR(Scalar nu = Scalar{0.5},
                   Scalar C  = Scalar{1.0},
                   Kernel kernel = Kernel::RBF,
                   int degree = 3,
                   Scalar gamma = Scalar{0},
                   Scalar coef0 = Scalar{0},
                   Scalar tol = Scalar{1e-3},
                   int max_iter = 1000,
                   std::optional<uint64_t> random_state = std::nullopt)
        : nu_(nu), C_(C), kernel_(kernel), degree_(degree),
          gamma_(gamma), coef0_(coef0),
          tol_(tol), max_iter_(max_iter),
          random_state_(random_state) {}

    [[nodiscard]] Scalar nu() const noexcept { return nu_; }
    [[nodiscard]] Scalar C() const noexcept { return C_; }
    [[nodiscard]] Kernel kernel() const noexcept { return kernel_; }

    [[nodiscard]] Scalar epsilon_fitted() const {
        this->check_is_fitted(); return epsilon_eff_;
    }
    [[nodiscard]] const std::vector<Eigen::Index>& support() const {
        this->check_is_fitted(); return support_;
    }
    [[nodiscard]] int n_support() const {
        this->check_is_fitted(); return static_cast<int>(support_.size());
    }

    SKIGEN_PARAMS(
        (nu,       nu_,       double),
        (C,        C_,        double),
        (degree,   degree_,   int),
        (gamma,    gamma_,    double),
        (coef0,    coef0_,    double),
        (tol,      tol_,      double),
        (max_iter, max_iter_, int))

    NuSVR& fit_impl(const Eigen::Ref<const MatrixType>& X,
                    const Eigen::Ref<const VectorType>& y) {
        internal::check_non_empty(X);
        internal::check_consistent_length(X, y);
        if (!(nu_ > Scalar{0} && nu_ <= Scalar{1})) {
            throw std::invalid_argument("NuSVR: nu must be in (0, 1].");
        }
        this->n_features_in_ = X.cols();

        const Eigen::Index n = X.rows();
        const Eigen::Index p = X.cols();

        const Scalar var_X = X.array().square().sum() /
                             static_cast<Scalar>(n * p) -
                             std::pow(X.mean(), Scalar{2});
        gamma_eff_ = (gamma_ > Scalar{0})
            ? gamma_
            : (var_X > Scalar{0}
                ? Scalar{1} / (static_cast<Scalar>(p) * var_X)
                : Scalar{1});

        MatrixType K(n, n);
        for (Eigen::Index i = 0; i < n; ++i) {
            for (Eigen::Index j = i; j < n; ++j) {
                const Scalar k = internal::kernel_eval<Scalar>(
                    kernel_, X.row(i), X.row(j), gamma_eff_, coef0_, degree_);
                K(i, j) = k;
                K(j, i) = k;
            }
        }

        // Two-phase fit: an initial epsilon-insensitive pass to estimate the
        // residual scale, then re-fit with epsilon set to the (1 - nu)
        // quantile of |residual| so a nu fraction lies outside the tube.
        VectorType beta;
        Scalar b;
        run_subgradient(K, y, Scalar{0}, beta, b);
        VectorType residual(n);
        for (Eigen::Index i = 0; i < n; ++i) {
            Scalar pred = b;
            for (Eigen::Index j = 0; j < n; ++j) pred += beta(j) * K(i, j);
            residual(i) = std::abs(y(i) - pred);
        }
        epsilon_eff_ = quantile(residual, Scalar{1} - nu_);
        run_subgradient(K, y, epsilon_eff_, beta, b);

        support_.clear();
        std::vector<Scalar> dual;
        for (Eigen::Index i = 0; i < n; ++i) {
            if (std::abs(beta(i)) > Scalar{1e-7}) {
                support_.push_back(i);
                dual.push_back(beta(i));
            }
        }
        const Eigen::Index ns = static_cast<Eigen::Index>(support_.size());
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
        const VectorType yh = predict_impl(X);
        const Scalar ym = y.mean();
        const Scalar ss_res = (y - yh).squaredNorm();
        const Scalar ss_tot = (y.array() - ym).matrix().squaredNorm();
        if (ss_tot == Scalar{0}) return Scalar{0};
        return Scalar{1} - ss_res / ss_tot;
    }

private:
    void run_subgradient(const MatrixType& K, const Eigen::Ref<const VectorType>& y,
                         Scalar epsilon, VectorType& beta, Scalar& b) const {
        const Eigen::Index n = K.rows();
        beta = VectorType::Zero(n);
        b = Scalar{0};
        std::mt19937_64 rng(random_state_.value_or(static_cast<uint64_t>(0)));
        std::vector<Eigen::Index> indices(static_cast<std::size_t>(n));
        std::iota(indices.begin(), indices.end(), Eigen::Index{0});
        const Scalar eta0 = Scalar{0.05};
        for (int t = 0; t < max_iter_; ++t) {
            std::shuffle(indices.begin(), indices.end(), rng);
            const Scalar eta = eta0 /
                (Scalar{1} + Scalar{0.001} * static_cast<Scalar>(t));
            Scalar total_update{0};
            for (auto i : indices) {
                Scalar pred = b;
                for (Eigen::Index j = 0; j < n; ++j) pred += beta(j) * K(i, j);
                const Scalar r = y(i) - pred;
                if (std::abs(r) <= epsilon) continue;
                const Scalar coeff = (r > Scalar{0}) ? C_ : -C_;
                Scalar new_b_i = std::clamp(beta(i) + eta * coeff, -C_, C_);
                const Scalar delta = new_b_i - beta(i);
                beta(i) = new_b_i;
                b += eta * coeff;
                total_update += std::abs(delta);
            }
            if (total_update / static_cast<Scalar>(n) < tol_) break;
        }
    }

    [[nodiscard]] static Scalar quantile(VectorType v, Scalar q) {
        std::sort(v.data(), v.data() + v.size());
        if (v.size() == 1) return v(0);
        const Scalar rank = std::clamp(q, Scalar{0}, Scalar{1}) *
                            static_cast<Scalar>(v.size() - 1);
        const Eigen::Index lo = static_cast<Eigen::Index>(std::floor(rank));
        const Eigen::Index hi = std::min(lo + 1, v.size() - 1);
        const Scalar f = rank - static_cast<Scalar>(lo);
        return v(lo) * (Scalar{1} - f) + v(hi) * f;
    }

    Scalar nu_;
    Scalar C_;
    Kernel kernel_;
    int degree_;
    Scalar gamma_;
    Scalar coef0_;
    Scalar tol_;
    int max_iter_;
    std::optional<uint64_t> random_state_;

    Scalar gamma_eff_ = Scalar{1};
    Scalar epsilon_eff_ = Scalar{0};
    std::vector<Eigen::Index> support_;
    MatrixType sv_X_;
    VectorType dual_coef_;
    Scalar intercept_ = Scalar{0};
};

/// @}

}  // namespace Skigen

#endif  // SKIGEN_SVM_NU_SVR_H
