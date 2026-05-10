// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_SVM_LINEAR_SVC_H
#define SKIGEN_SVM_LINEAR_SVC_H

#include "../Core/Base.h"
#include "../Core/Validation.h"

#include <Eigen/Core>
#include <algorithm>
#include <cstdint>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace Skigen {

/// @defgroup Algo_LinearSVM Linear Support Vector Machines
/// @ingroup SVM
/// @brief Liblinear-style coordinate descent for linear SVMs.
/// @{

/// @brief Linear SVC — primal sub-gradient SGD on the squared-hinge L2
///   regularised objective.
///
/// Mirrors the binary case of
/// [sklearn.svm.LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html);
/// multiclass uses one-vs-rest (default `multi_class="ovr"`).
///
/// Solves @f$ \min_w \tfrac{1}{2}\|w\|^2 + C \sum_i \max(0, 1 - y_i (w \cdot x_i + b))^2 @f$
/// (`loss="squared_hinge"`, the sklearn default) or the analogous hinge
/// loss when `loss="hinge"`.
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default |
/// |---|---|---|
/// | `C`             | `Scalar`              | `1.0` |
/// | `loss`          | `Loss`                | `SquaredHinge` |
/// | `tol`           | `Scalar`              | `1e-4` |
/// | `max_iter`      | `int`                 | `1000` |
/// | `fit_intercept` | `bool`                | `true` |
/// | `random_state`  | `optional<uint64_t>`  | `nullopt` |
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type |
/// |---|---|
/// | `classes()`     | `Eigen::VectorXi` |
/// | `n_classes()`   | `int` |
/// | `coef()`        | `MatrixType` (n_classes_ovr × n_features) |
/// | `intercept()`   | `VectorType` (n_classes_ovr,) |
///
/// ### Limitations relative to scikit-learn
///
/// The solver is a primal sub-gradient SGD over the regularised hinge /
/// squared-hinge objective rather than liblinear's coordinate-descent
/// dual solver — converged solutions agree at the optimum but the
/// iteration trace differs. `penalty="l1"`, `dual=true`,
/// `class_weight`, `intercept_scaling`, and the
/// `multi_class="crammer_singer"` joint multiclass formulation are not
/// implemented.
template <typename Scalar = double>
class LinearSVC : public Classifier<LinearSVC<Scalar>, Scalar> {
public:
    using Base = Classifier<LinearSVC<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;
    using typename Base::ScalarType;
    using typename Base::LabelType;
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    enum class Loss { Hinge, SquaredHinge };

    explicit LinearSVC(Scalar C = Scalar{1.0},
                       Loss loss = Loss::SquaredHinge,
                       Scalar tol = Scalar{1e-4},
                       int max_iter = 1000,
                       bool fit_intercept = true,
                       std::optional<uint64_t> random_state = std::nullopt)
        : C_(C), loss_(loss), tol_(tol),
          max_iter_(max_iter),
          fit_intercept_(fit_intercept),
          random_state_(random_state) {}

    // -- Accessors ----------------------------------------------------------

    [[nodiscard]] Scalar C()    const noexcept { return C_; }
    [[nodiscard]] Loss   loss() const noexcept { return loss_; }

    [[nodiscard]] const Eigen::VectorXi& classes() const {
        this->check_is_fitted(); return classes_;
    }
    [[nodiscard]] int n_classes() const {
        this->check_is_fitted(); return static_cast<int>(classes_.size());
    }
    [[nodiscard]] const MatrixType& coef() const {
        this->check_is_fitted(); return coef_;
    }
    [[nodiscard]] const VectorType& intercept() const {
        this->check_is_fitted(); return intercept_;
    }

    // -- Fit / Predict ------------------------------------------------------

    LinearSVC& fit_impl(const Eigen::Ref<const MatrixType>& X,
                        const Eigen::Ref<const Eigen::VectorXi>& y) {
        internal::check_non_empty(X);
        internal::check_consistent_length(X, y);
        this->n_features_in_ = X.cols();

        // Discover sorted unique labels.
        std::vector<int> uniq;
        uniq.reserve(static_cast<std::size_t>(y.size()));
        for (Eigen::Index i = 0; i < y.size(); ++i) uniq.push_back(y(i));
        std::sort(uniq.begin(), uniq.end());
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
        const int K = static_cast<int>(uniq.size());
        if (K < 2) {
            throw std::invalid_argument(
                "LinearSVC.fit: at least 2 distinct classes required.");
        }
        classes_ = Eigen::VectorXi(K);
        for (int i = 0; i < K; ++i) classes_(i) = uniq[i];

        const Eigen::Index n = X.rows();
        const Eigen::Index p = X.cols();

        // Binary: 1 model with y ∈ {-1, +1}; multiclass: K One-vs-Rest.
        const int n_models = (K == 2) ? 1 : K;
        coef_      = MatrixType::Zero(n_models, p);
        intercept_ = VectorType::Zero(n_models);

        for (int m = 0; m < n_models; ++m) {
            VectorType y01(n);
            const int pos_label_idx = (K == 2) ? 1 : m;
            for (Eigen::Index i = 0; i < n; ++i) {
                int lbl_idx = -1;
                for (int kk = 0; kk < K; ++kk) {
                    if (uniq[kk] == y(i)) { lbl_idx = kk; break; }
                }
                y01(i) = (lbl_idx == pos_label_idx)
                    ? Scalar{1} : Scalar{-1};
            }

            RowVectorType w = RowVectorType::Zero(p);
            Scalar b{0};
            fit_one(X, y01, w, b);
            coef_.row(m) = w;
            intercept_(m) = b;
        }

        this->fitted_ = true;
        return *this;
    }

    [[nodiscard]] LabelType predict_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        MatrixType df = decision_function(X);
        LabelType out(X.rows());
        if (df.cols() == 1) {
            // Binary path: column 0 is +1's score.
            for (Eigen::Index i = 0; i < X.rows(); ++i) {
                out(i) = (df(i, 0) >= Scalar{0})
                    ? classes_(1) : classes_(0);
            }
        } else {
            for (Eigen::Index i = 0; i < X.rows(); ++i) {
                Eigen::Index k;
                df.row(i).maxCoeff(&k);
                out(i) = classes_(k);
            }
        }
        return out;
    }

    /// @brief Raw decision function. Shape is (n_samples,) for binary
    ///   problems wrapped as a 1-column matrix; (n_samples, n_classes)
    ///   for multiclass.
    [[nodiscard]] MatrixType decision_function(
        const Eigen::Ref<const MatrixType>& X) const {
        this->check_is_fitted();
        MatrixType df = X * coef_.transpose();
        df.rowwise() += intercept_.transpose();
        return df;
    }

private:
    void fit_one(const Eigen::Ref<const MatrixType>& X,
                 const VectorType& y01,
                 RowVectorType& w, Scalar& b) const {
        const Eigen::Index n = X.rows();
        const Eigen::Index p = X.cols();
        const uint64_t seed =
            random_state_.value_or(static_cast<uint64_t>(42));
        std::mt19937_64 rng(seed);

        std::vector<Eigen::Index> indices(static_cast<std::size_t>(n));
        std::iota(indices.begin(), indices.end(), Eigen::Index{0});

        const Scalar alpha = Scalar{1} /
                             (C_ * static_cast<Scalar>(n));
        // Inverse-scaling LR.
        for (int t = 0; t < max_iter_; ++t) {
            std::shuffle(indices.begin(), indices.end(), rng);
            const Scalar eta = Scalar{1} /
                (Scalar{1} + alpha * static_cast<Scalar>(t));
            Scalar total_update{0};
            for (auto idx : indices) {
                const Scalar margin =
                    (X.row(idx) * w.transpose())(0) + b;
                const Scalar yi = y01(idx);
                if (loss_ == Loss::Hinge) {
                    if (yi * margin < Scalar{1}) {
                        w += eta * (C_ * yi * X.row(idx) - alpha * w);
                        if (fit_intercept_) b += eta * C_ * yi;
                        total_update += std::abs(eta * C_ * yi);
                    } else {
                        w *= (Scalar{1} - eta * alpha);
                    }
                } else {  // SquaredHinge
                    const Scalar slack =
                        std::max(Scalar{0}, Scalar{1} - yi * margin);
                    if (slack > Scalar{0}) {
                        const Scalar coeff = Scalar{2} * C_ * slack * yi;
                        w += eta * (coeff * X.row(idx) - alpha * w);
                        if (fit_intercept_) b += eta * coeff;
                        total_update += std::abs(eta * coeff);
                    } else {
                        w *= (Scalar{1} - eta * alpha);
                    }
                }
            }
            (void)p;
            if (total_update / static_cast<Scalar>(n) < tol_) break;
        }
    }

    Scalar C_;
    Loss loss_;
    Scalar tol_;
    int max_iter_;
    bool fit_intercept_;
    std::optional<uint64_t> random_state_;

    Eigen::VectorXi classes_;
    MatrixType coef_;
    VectorType intercept_;
};

/// @}

}  // namespace Skigen

#endif  // SKIGEN_SVM_LINEAR_SVC_H
