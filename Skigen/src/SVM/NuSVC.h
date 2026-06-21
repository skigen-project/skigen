// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

#ifndef SKIGEN_SVM_NU_SVC_H
#define SKIGEN_SVM_NU_SVC_H

#include "../Core/Base.h"
#include "../Core/Validation.h"
#include "Detail/Kernels.h"
#include "Detail/SMO.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace Skigen {

/// @addtogroup Algo_KernelSVM
/// @{

/// @brief Nu-Support Vector Classification with kernels.
///
/// Solves the nu-SVM binary classification dual via a dedicated SMO variant.
/// The `nu` parameter is an upper bound on the fraction of margin errors and
/// a lower bound on the fraction of support vectors.
///
/// Mirrors the dense binary core of
/// [sklearn.svm.NuSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html).
template <typename Scalar = double>
class NuSVC : public Classifier<NuSVC<Scalar>, Scalar> {
public:
    using Base = Classifier<NuSVC<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::IndexType;
    using typename Base::ScalarType;
    using typename Base::LabelType;
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Kernel = internal::KernelKind;
    using Base::fit;
    using Base::predict;

    explicit NuSVC(Scalar nu = Scalar{0.5},
                   Kernel kernel = Kernel::RBF,
                   int degree = 3,
                   Scalar gamma = Scalar{0},
                   Scalar coef0 = Scalar{0},
                   Scalar tol = Scalar{1e-3},
                   int max_passes = 50,
                   std::optional<uint64_t> random_state = std::nullopt)
        : nu_(nu), kernel_(kernel), degree_(degree),
          gamma_(gamma), coef0_(coef0),
          tol_(tol), max_passes_(max_passes),
          random_state_(random_state) {}

    [[nodiscard]] Scalar nu() const noexcept { return nu_; }
    [[nodiscard]] Kernel kernel() const noexcept { return kernel_; }

    [[nodiscard]] const Eigen::VectorXi& classes() const {
        this->check_is_fitted(); return classes_;
    }
    [[nodiscard]] const std::vector<Eigen::Index>& support() const {
        this->check_is_fitted(); return support_;
    }
    [[nodiscard]] int n_support() const {
        this->check_is_fitted(); return static_cast<int>(support_.size());
    }
    [[nodiscard]] Scalar intercept() const {
        this->check_is_fitted(); return intercept_;
    }

    SKIGEN_PARAMS(
        (nu,         nu_,         double),
        (degree,     degree_,     int),
        (gamma,      gamma_,      double),
        (coef0,      coef0_,      double),
        (tol,        tol_,        double),
        (max_passes, max_passes_, int))

    NuSVC& fit_impl(const Eigen::Ref<const MatrixType>& X,
                    const Eigen::Ref<const Eigen::VectorXi>& y) {
        internal::check_non_empty(X);
        internal::check_consistent_length(X, y);
        if (!(nu_ > Scalar{0} && nu_ <= Scalar{1})) {
            throw std::invalid_argument("NuSVC: nu must be in (0, 1].");
        }
        this->n_features_in_ = X.cols();

        std::vector<int> uniq;
        uniq.reserve(static_cast<std::size_t>(y.size()));
        for (Eigen::Index i = 0; i < y.size(); ++i) uniq.push_back(y(i));
        std::sort(uniq.begin(), uniq.end());
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
        if (uniq.size() != 2) {
            throw std::invalid_argument(
                "NuSVC: only binary classification is supported; got " +
                std::to_string(uniq.size()) + " classes.");
        }
        classes_ = Eigen::VectorXi(2);
        classes_(0) = uniq[0];
        classes_(1) = uniq[1];

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

        VectorType y_pm(n);
        for (Eigen::Index i = 0; i < n; ++i) {
            y_pm(i) = (y(i) == classes_(1)) ? Scalar{1} : Scalar{-1};
        }

        MatrixType K(n, n);
        for (Eigen::Index i = 0; i < n; ++i) {
            for (Eigen::Index j = i; j < n; ++j) {
                const Scalar k = internal::kernel_eval<Scalar>(
                    kernel_, X.row(i), X.row(j), gamma_eff_, coef0_, degree_);
                K(i, j) = k;
                K(j, i) = k;
            }
        }

        VectorType alpha;
        Scalar b;
        const bool ok = internal::smo_nu_classification<Scalar>(
            y_pm, K, nu_, tol_, max_passes_, random_state_, alpha, b);
        if (!ok) {
            throw std::invalid_argument(
                "NuSVC: nu is infeasible for the given class balance; "
                "reduce nu.");
        }

        support_.clear();
        std::vector<Scalar> coefs;
        for (Eigen::Index i = 0; i < n; ++i) {
            if (alpha(i) > Scalar{1e-7}) {
                support_.push_back(i);
                coefs.push_back(alpha(i) * y_pm(i));
            }
        }
        const Eigen::Index ns = static_cast<Eigen::Index>(support_.size());
        sv_X_ = MatrixType(ns, p);
        dual_coef_ = VectorType(ns);
        for (Eigen::Index s = 0; s < ns; ++s) {
            sv_X_.row(s) = X.row(support_[static_cast<std::size_t>(s)]);
            dual_coef_(s) = coefs[static_cast<std::size_t>(s)];
        }
        intercept_ = b;

        this->fitted_ = true;
        return *this;
    }

    [[nodiscard]] LabelType predict_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        const VectorType df = decision_function(X);
        LabelType out(X.rows());
        for (Eigen::Index i = 0; i < X.rows(); ++i) {
            out(i) = (df(i) >= Scalar{0}) ? classes_(1) : classes_(0);
        }
        return out;
    }

    /// @brief Raw decision function — sum_s dual_coef_s K(x, sv_s) + b.
    [[nodiscard]] VectorType decision_function(
        const Eigen::Ref<const MatrixType>& X) const {
        this->check_is_fitted();
        this->validate_feature_count(X);
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

private:
    Scalar nu_;
    Kernel kernel_;
    int degree_;
    Scalar gamma_;
    Scalar coef0_;
    Scalar tol_;
    int max_passes_;
    std::optional<uint64_t> random_state_;

    Scalar gamma_eff_ = Scalar{0};
    Eigen::VectorXi classes_;
    std::vector<Eigen::Index> support_;
    MatrixType sv_X_;
    VectorType dual_coef_;
    Scalar intercept_ = Scalar{0};
};

/// @}

}  // namespace Skigen

#endif  // SKIGEN_SVM_NU_SVC_H
