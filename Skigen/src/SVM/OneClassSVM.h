// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

#ifndef SKIGEN_SVM_ONE_CLASS_SVM_H
#define SKIGEN_SVM_ONE_CLASS_SVM_H

#include "../Core/Base.h"
#include "../Core/Validation.h"
#include "Detail/Kernels.h"
#include "Detail/SMO.h"

#include <Eigen/Core>

#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

namespace Skigen {

/// @addtogroup Algo_KernelSVM
/// @{

/// @brief Unsupervised outlier detection with a one-class SVM.
///
/// Solves the one-class nu-SVM dual with a dedicated SMO variant and exposes
/// sklearn-style scoring: `decision_function`, `score_samples`, `predict`
/// (`+1` inlier, `-1` outlier), and `offset_`.
///
/// Mirrors the dense core of
/// [sklearn.svm.OneClassSVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html).
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `kernel` | `Kernel` | `RBF` | Kernel function. |
/// | `degree` | `int` | `3` | Polynomial-kernel degree. |
/// | `gamma` | `Scalar` | `0` | Kernel coefficient; `0` means `1 / (n_features * var)`. |
/// | `coef0` | `Scalar` | `0` | Independent term for poly / sigmoid kernels. |
/// | `nu` | `Scalar` | `0.5` | Upper bound on outlier fraction, in `(0, 1]`. |
/// | `tol` | `Scalar` | `1e-3` | Solver tolerance. |
/// | `max_passes` | `int` | `50` | Solver passes without change before stopping. |
///
/// ### Examples
///
/// @snippet one_class_svm.cpp example_one_class_svm
template <typename Scalar = double>
class OneClassSVM : public Estimator<OneClassSVM<Scalar>, Scalar> {
public:
    using Base = Estimator<OneClassSVM<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::IndexType;
    using typename Base::ScalarType;
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Kernel = internal::KernelKind;

    explicit OneClassSVM(Kernel kernel = Kernel::RBF,
                         int degree = 3,
                         Scalar gamma = Scalar{0},
                         Scalar coef0 = Scalar{0},
                         Scalar nu = Scalar{0.5},
                         Scalar tol = Scalar{1e-3},
                         int max_passes = 50,
                         std::optional<uint64_t> random_state = std::nullopt)
        : kernel_(kernel), degree_(degree),
          gamma_(gamma), coef0_(coef0), nu_(nu),
          tol_(tol), max_passes_(max_passes),
          random_state_(random_state) {}

    [[nodiscard]] Kernel kernel() const noexcept { return kernel_; }
    [[nodiscard]] Scalar gamma() const noexcept { return gamma_; }
    [[nodiscard]] Scalar nu() const noexcept { return nu_; }

    [[nodiscard]] const std::vector<Eigen::Index>& support() const {
        this->check_is_fitted(); return support_;
    }
    [[nodiscard]] int n_support() const {
        this->check_is_fitted(); return static_cast<int>(support_.size());
    }
    [[nodiscard]] const VectorType& dual_coef() const {
        this->check_is_fitted(); return dual_coef_;
    }
    [[nodiscard]] Scalar offset() const {
        this->check_is_fitted(); return offset_;
    }

    SKIGEN_PARAMS(
        (degree,     degree_,     int),
        (gamma,      gamma_,      double),
        (coef0,      coef0_,      double),
        (nu,         nu_,         double),
        (tol,        tol_,        double),
        (max_passes, max_passes_, int))

    OneClassSVM& fit(const Eigen::Ref<const MatrixType>& X) {
        internal::check_non_empty(X);
        if (!(nu_ > Scalar{0} && nu_ <= Scalar{1})) {
            throw std::invalid_argument("OneClassSVM: nu must be in (0, 1].");
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

        VectorType alpha;
        Scalar rho;
        internal::smo_one_class<Scalar>(
            K, nu_, tol_, max_passes_, random_state_, alpha, rho);

        support_.clear();
        std::vector<Scalar> coefs;
        for (Eigen::Index i = 0; i < n; ++i) {
            if (alpha(i) > Scalar{1e-8}) {
                support_.push_back(i);
                coefs.push_back(alpha(i));
            }
        }
        const Eigen::Index ns = static_cast<Eigen::Index>(support_.size());
        sv_X_ = MatrixType(ns, p);
        dual_coef_ = VectorType(ns);
        for (Eigen::Index s = 0; s < ns; ++s) {
            sv_X_.row(s) = X.row(support_[static_cast<std::size_t>(s)]);
            dual_coef_(s) = coefs[static_cast<std::size_t>(s)];
        }
        // sklearn stores offset_ = -rho; decision_function = score - rho.
        offset_ = -rho;

        this->fitted_ = true;
        return *this;
    }

    /// @brief Signed distance to the separating hyperplane (rho-shifted).
    [[nodiscard]] VectorType decision_function(
        const Eigen::Ref<const MatrixType>& X) const {
        this->check_is_fitted();
        this->validate_feature_count(X);
        return (raw_scores(X).array() + offset_).matrix();
    }

    /// @brief Unshifted kernel score for each sample.
    [[nodiscard]] VectorType score_samples(
        const Eigen::Ref<const MatrixType>& X) const {
        this->check_is_fitted();
        this->validate_feature_count(X);
        return raw_scores(X);
    }

    /// @brief Predict `+1` for inliers and `-1` for outliers.
    [[nodiscard]] Eigen::VectorXi predict(
        const Eigen::Ref<const MatrixType>& X) const {
        const VectorType decisions = decision_function(X);
        Eigen::VectorXi labels(X.rows());
        for (Eigen::Index i = 0; i < X.rows(); ++i) {
            labels(i) = decisions(i) >= Scalar{0} ? 1 : -1;
        }
        return labels;
    }

private:
    [[nodiscard]] VectorType raw_scores(
        const Eigen::Ref<const MatrixType>& X) const {
        VectorType out(X.rows());
        for (Eigen::Index i = 0; i < X.rows(); ++i) {
            Scalar s{0};
            for (Eigen::Index t = 0; t < dual_coef_.size(); ++t) {
                s += dual_coef_(t) * internal::kernel_eval<Scalar>(
                    kernel_, X.row(i), sv_X_.row(t),
                    gamma_eff_, coef0_, degree_);
            }
            out(i) = s;
        }
        return out;
    }

    Kernel kernel_;
    int degree_;
    Scalar gamma_;
    Scalar coef0_;
    Scalar nu_;
    Scalar tol_;
    int max_passes_;
    std::optional<uint64_t> random_state_;

    Scalar gamma_eff_ = Scalar{0};
    std::vector<Eigen::Index> support_;
    MatrixType sv_X_;
    VectorType dual_coef_;
    Scalar offset_ = Scalar{0};
};

/// @}

}  // namespace Skigen

#endif  // SKIGEN_SVM_ONE_CLASS_SVM_H
