// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_SVM_SVC_H
#define SKIGEN_SVM_SVC_H

#include "../Core/Base.h"
#include "../Core/Validation.h"
#include "../Calibration/PlattScaling.h"
#include "Detail/Kernels.h"
#include "Detail/SMO.h"

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace Skigen {

/// @defgroup Algo_KernelSVM Kernel Support Vector Machines
/// @ingroup SVM
/// @brief libsvm-style C-SVC / SVR with kernels.
/// @{

/// @brief C-Support Vector Classification with kernels.
///
/// Mirrors
/// [sklearn.svm.SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
/// for the binary case (multiclass via one-vs-one as in libsvm).
///
/// When `probability=true`, Platt scaling is fitted via internal 5-fold
/// cross-validation on the decision function scores, providing calibrated
/// posterior probabilities through `predict_proba()`.
template <typename Scalar = double>
class SVC : public Classifier<SVC<Scalar>, Scalar> {
public:
    using Base = Classifier<SVC<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;
    using typename Base::ScalarType;
    using typename Base::LabelType;
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    using Kernel = internal::KernelKind;
    using Base::fit;
    using Base::predict;

    explicit SVC(Scalar C = Scalar{1.0},
                 Kernel kernel = Kernel::RBF,
                 int degree = 3,
                 Scalar gamma = Scalar{0},
                 Scalar coef0 = Scalar{0},
                 bool probability = false,
                 Scalar tol = Scalar{1e-3},
                 int max_passes = 50,
                 std::optional<uint64_t> random_state = std::nullopt)
        : C_(C), kernel_(kernel), degree_(degree),
          gamma_(gamma), coef0_(coef0),
          probability_(probability),
          tol_(tol), max_passes_(max_passes),
          random_state_(random_state) {}

    [[nodiscard]] Scalar  C()       const noexcept { return C_; }
    [[nodiscard]] Kernel  kernel()  const noexcept { return kernel_; }
    [[nodiscard]] Scalar  gamma()   const noexcept { return gamma_; }

    [[nodiscard]] const Eigen::VectorXi& classes() const {
        this->check_is_fitted(); return classes_;
    }
    [[nodiscard]] int n_classes() const {
        this->check_is_fitted(); return static_cast<int>(classes_.size());
    }
    [[nodiscard]] const std::vector<Eigen::Index>& support() const {
        this->check_is_fitted(); return support_;
    }
    [[nodiscard]] int n_support() const {
        this->check_is_fitted();
        return static_cast<int>(support_.size());
    }

    SKIGEN_PARAMS(
        (C,          C_,          double),
        (degree,     degree_,     int),
        (gamma,      gamma_,     double),
        (coef0,      coef0_,     double),
        (probability,probability_,bool),
        (tol,        tol_,       double),
        (max_passes, max_passes_, int))

    SVC& fit_impl(const Eigen::Ref<const MatrixType>& X,
                  const Eigen::Ref<const Eigen::VectorXi>& y) {
        internal::check_non_empty(X);
        internal::check_consistent_length(X, y);
        this->n_features_in_ = X.cols();

        std::vector<int> uniq;
        uniq.reserve(static_cast<std::size_t>(y.size()));
        for (Eigen::Index i = 0; i < y.size(); ++i) uniq.push_back(y(i));
        std::sort(uniq.begin(), uniq.end());
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
        if (uniq.size() != 2) {
            throw std::invalid_argument(
                "SVC: only binary classification is supported; got " +
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
                    kernel_, X.row(i), X.row(j),
                    gamma_eff_, coef0_, degree_);
                K(i, j) = k;
                K(j, i) = k;
            }
        }

        VectorType alpha;
        Scalar b;
        internal::smo_binary<Scalar>(
            y_pm, K, C_, tol_, max_passes_, random_state_, alpha, b);

        support_.clear();
        std::vector<Scalar> alpha_y;
        for (Eigen::Index i = 0; i < n; ++i) {
            if (alpha(i) > Scalar{1e-7}) {
                support_.push_back(i);
                alpha_y.push_back(alpha(i) * y_pm(i));
            }
        }
        const Eigen::Index ns = static_cast<Eigen::Index>(support_.size());
        sv_X_ = MatrixType(ns, p);
        dual_coef_ = VectorType(ns);
        for (Eigen::Index s = 0; s < ns; ++s) {
            sv_X_.row(s) = X.row(support_[static_cast<std::size_t>(s)]);
            dual_coef_(s) =
                alpha_y[static_cast<std::size_t>(s)];
        }
        intercept_ = b;

        if (probability_) {
            fit_platt_cv(X, y_pm);
        }

        this->fitted_ = true;
        return *this;
    }

    /// @brief Fit from a sparse design matrix (densifies internally).
    template <int Options, typename StorageIndex>
    SVC& fit(const Eigen::SparseMatrix<Scalar, Options, StorageIndex>& X,
             const Eigen::Ref<const Eigen::VectorXi>& y) {
        if (X.rows() == 0 || X.cols() == 0)
            throw std::invalid_argument("SVC.fit: empty sparse matrix.");
        MatrixType Xd = MatrixType(X);
        return fit_impl(Xd, y);
    }

    [[nodiscard]] LabelType predict_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        VectorType df = decision_function(X);
        LabelType out(X.rows());
        for (Eigen::Index i = 0; i < X.rows(); ++i) {
            out(i) = (df(i) >= Scalar{0}) ? classes_(1) : classes_(0);
        }
        return out;
    }

    /// @brief Predict from a sparse design matrix (densifies internally).
    template <int Options, typename StorageIndex>
    [[nodiscard]] LabelType predict(
        const Eigen::SparseMatrix<Scalar, Options, StorageIndex>& X) const {
        this->check_is_fitted();
        MatrixType Xd = MatrixType(X);
        return predict_impl(Xd);
    }

    /// @brief Raw decision function — sum_s alpha_s y_s K(x, sv_s) + b.
    [[nodiscard]] VectorType decision_function(
        const Eigen::Ref<const MatrixType>& X) const {
        this->check_is_fitted();
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

    /// @brief Calibrated posterior probabilities via Platt scaling.
    ///
    /// Requires `probability=true` at construction time.
    /// Returns a (n_samples, 2) matrix with columns [P(y=class_0), P(y=class_1)].
    [[nodiscard]] MatrixType predict_proba(
        const Eigen::Ref<const MatrixType>& X) const {
        this->check_is_fitted();
        if (!probability_) {
            throw std::runtime_error(
                "SVC.predict_proba: requires probability=true.");
        }
        VectorType df = decision_function(X);
        VectorType p_pos = internal::apply_platt_sigmoid<Scalar>(
            df, platt_A_, platt_B_);
        MatrixType P(X.rows(), 2);
        P.col(1) = p_pos;
        P.col(0) = (Scalar{1} - p_pos.array()).matrix();
        return P;
    }

private:
    void fit_platt_cv(const Eigen::Ref<const MatrixType>& X,
                      const VectorType& y_pm) {
        const Eigen::Index n = X.rows();
        const int cv = 5;
        const Eigen::Index fold_size = n / cv;

        VectorType oof_scores(n);
        VectorType oof_targets(n);

        std::vector<Eigen::Index> indices(static_cast<std::size_t>(n));
        std::iota(indices.begin(), indices.end(), Eigen::Index{0});
        const uint64_t seed = random_state_.value_or(static_cast<uint64_t>(0));
        std::mt19937_64 rng(seed);
        std::shuffle(indices.begin(), indices.end(), rng);

        for (int f = 0; f < cv; ++f) {
            const Eigen::Index val_start = f * fold_size;
            const Eigen::Index val_end =
                (f == cv - 1) ? n : (f + 1) * fold_size;
            const Eigen::Index n_val = val_end - val_start;
            const Eigen::Index n_trn = n - n_val;

            MatrixType X_trn(n_trn, X.cols());
            VectorType y_trn(n_trn);
            MatrixType X_val(n_val, X.cols());
            Eigen::Index ti = 0;
            for (Eigen::Index i = 0; i < n; ++i) {
                const Eigen::Index idx = indices[static_cast<std::size_t>(i)];
                if (i >= val_start && i < val_end) {
                    X_val.row(i - val_start) = X.row(idx);
                    oof_targets(i) = (y_pm(idx) > Scalar{0})
                        ? Scalar{1} : Scalar{0};
                } else {
                    X_trn.row(ti) = X.row(idx);
                    y_trn(ti) = y_pm(idx);
                    ++ti;
                }
            }

            MatrixType Kf(n_trn, n_trn);
            for (Eigen::Index i = 0; i < n_trn; ++i)
                for (Eigen::Index j = i; j < n_trn; ++j) {
                    const Scalar k = internal::kernel_eval<Scalar>(
                        kernel_, X_trn.row(i), X_trn.row(j),
                        gamma_eff_, coef0_, degree_);
                    Kf(i, j) = k; Kf(j, i) = k;
                }

            VectorType alpha_f;
            Scalar b_f;
            internal::smo_binary<Scalar>(
                y_trn, Kf, C_, tol_, max_passes_, random_state_,
                alpha_f, b_f);

            for (Eigen::Index vi = 0; vi < n_val; ++vi) {
                Scalar s{0};
                for (Eigen::Index t = 0; t < n_trn; ++t) {
                    if (alpha_f(t) < Scalar{1e-7}) continue;
                    s += alpha_f(t) * y_trn(t) *
                        internal::kernel_eval<Scalar>(
                            kernel_, X_val.row(vi), X_trn.row(t),
                            gamma_eff_, coef0_, degree_);
                }
                oof_scores(val_start + vi) = s + b_f;
            }
        }

        internal::fit_platt_sigmoid<Scalar>(
            oof_scores, oof_targets, platt_A_, platt_B_);
    }

    Scalar C_;
    Kernel kernel_;
    int degree_;
    Scalar gamma_;
    Scalar coef0_;
    bool probability_;
    Scalar tol_;
    int max_passes_;
    std::optional<uint64_t> random_state_;

    Scalar gamma_eff_{1};
    Eigen::VectorXi classes_;
    std::vector<Eigen::Index> support_;
    MatrixType sv_X_;
    VectorType dual_coef_;
    Scalar intercept_{0};
    Scalar platt_A_{0};
    Scalar platt_B_{0};
};

/// @}

}  // namespace Skigen

#endif  // SKIGEN_SVM_SVC_H
