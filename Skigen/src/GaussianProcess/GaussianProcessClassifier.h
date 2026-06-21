// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

#ifndef SKIGEN_GAUSSIAN_PROCESS_CLASSIFIER_H
#define SKIGEN_GAUSSIAN_PROCESS_CLASSIFIER_H

#include "Kernels.h"
#include "../Core/Base.h"
#include "../Core/Validation.h"

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace Skigen {

/// @addtogroup Algo_GaussianProcess
/// @{

/// @brief Dense binary Gaussian Process classifier.
///
/// Fits a fixed-kernel binary Gaussian Process classifier with a logistic
/// likelihood and Laplace posterior approximation. The implementation exposes
/// sklearn-style class labels, posterior latent scores, class probabilities,
/// and deterministic prediction semantics for dense numeric input.
///
/// Mirrors the dense fixed-hyperparameter binary subset of
/// `sklearn.gaussian_process.GaussianProcessClassifier`.
///
/// ### Examples
///
/// @snippet gaussian_process_classifier.cpp example_gaussian_process_classifier
template <typename Scalar = double>
class GaussianProcessClassifier
    : public Classifier<GaussianProcessClassifier<Scalar>, Scalar> {
public:
    using Base = Classifier<GaussianProcessClassifier<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::IndexType;
    using LabelType = typename Base::LabelType;
    using Kernel = gaussian_process::Kernel;

    explicit GaussianProcessClassifier(
        Kernel kernel = Kernel::RBF,
        Scalar alpha = Scalar{1e-10},
        Scalar length_scale = Scalar{1},
        Scalar constant_value = Scalar{1},
        Scalar noise_level = Scalar{1},
        Scalar nu = Scalar{1.5},
        Scalar rational_quadratic_alpha = Scalar{1},
        Scalar periodicity = Scalar{1},
        Scalar sigma_0 = Scalar{1},
        int max_iter = 100,
        Scalar tol = Scalar{1e-6})
        : kernel_(kernel),
          alpha_(alpha),
          length_scale_(length_scale),
          constant_value_(constant_value),
          noise_level_(noise_level),
          nu_(nu),
          rational_quadratic_alpha_(rational_quadratic_alpha),
          periodicity_(periodicity),
          sigma_0_(sigma_0),
          max_iter_(max_iter),
          tol_(tol) {}

    [[nodiscard]] Kernel kernel() const noexcept { return kernel_; }
    [[nodiscard]] Scalar alpha() const noexcept { return alpha_; }
    [[nodiscard]] Scalar length_scale() const noexcept { return length_scale_; }
    [[nodiscard]] Scalar constant_value() const noexcept { return constant_value_; }
    [[nodiscard]] Scalar noise_level() const noexcept { return noise_level_; }
    [[nodiscard]] Scalar nu() const noexcept { return nu_; }
    [[nodiscard]] Scalar rational_quadratic_alpha() const noexcept {
        return rational_quadratic_alpha_;
    }
    [[nodiscard]] Scalar periodicity() const noexcept { return periodicity_; }
    [[nodiscard]] Scalar sigma_0() const noexcept { return sigma_0_; }
    [[nodiscard]] int max_iter() const noexcept { return max_iter_; }
    [[nodiscard]] Scalar tol() const noexcept { return tol_; }

    [[nodiscard]] const LabelType& classes() const {
        this->check_is_fitted();
        return classes_;
    }

    [[nodiscard]] const MatrixType& X_train() const {
        this->check_is_fitted();
        return X_train_;
    }

    [[nodiscard]] const VectorType& latent_mean() const {
        this->check_is_fitted();
        return latent_mean_;
    }

    [[nodiscard]] const VectorType& dual_coef() const {
        this->check_is_fitted();
        return dual_coef_;
    }

    [[nodiscard]] int n_iter() const {
        this->check_is_fitted();
        return n_iter_;
    }

    SKIGEN_PARAMS(
        (alpha, alpha_, double),
        (length_scale, length_scale_, double),
        (constant_value, constant_value_, double),
        (noise_level, noise_level_, double),
        (nu, nu_, double),
        (rational_quadratic_alpha, rational_quadratic_alpha_, double),
        (periodicity, periodicity_, double),
        (sigma_0, sigma_0_, double),
        (max_iter, max_iter_, int),
        (tol, tol_, double))

    GaussianProcessClassifier& fit_impl(const Eigen::Ref<const MatrixType>& input,
                                        const Eigen::Ref<const LabelType>& target) {
        internal::check_non_empty(input);
        internal::check_consistent_length(input, target);
        validate_parameters();

        this->n_features_in_ = input.cols();
        X_train_ = input;
        encode_classes(target);

        kernel_train_ = kernel_matrix(X_train_, X_train_, /*same_matrix=*/true);
        kernel_train_.diagonal().array() += std::max(alpha_, Scalar{1e-12});

        VectorType f = VectorType::Zero(input.rows());
        VectorType y01(input.rows());
        for (IndexType row = 0; row < target.rows(); ++row) {
            y01(row) = target(row) == classes_(1) ? Scalar{1} : Scalar{0};
        }

        for (int iteration = 0; iteration < max_iter_; ++iteration) {
            VectorType alpha_candidate;
            VectorType sqrt_w_candidate;
            MatrixType l_candidate;
            const VectorType f_new = laplace_step(
                f, y01, alpha_candidate, sqrt_w_candidate, l_candidate);
            n_iter_ = iteration + 1;
            if ((f_new - f).cwiseAbs().maxCoeff() <= tol_) {
                f = f_new;
                dual_coef_ = std::move(alpha_candidate);
                w_sqrt_ = std::move(sqrt_w_candidate);
                L_ = std::move(l_candidate);
                break;
            }
            f = f_new;
            dual_coef_ = std::move(alpha_candidate);
            w_sqrt_ = std::move(sqrt_w_candidate);
            L_ = std::move(l_candidate);
        }

        latent_mean_ = kernel_train_ * dual_coef_;
        this->fitted_ = true;
        return *this;
    }

    [[nodiscard]] LabelType predict_impl(const Eigen::Ref<const MatrixType>& input) const {
        const MatrixType probabilities = predict_proba(input);
        LabelType labels(input.rows());
        for (IndexType row = 0; row < input.rows(); ++row) {
            labels(row) = probabilities(row, 1) >= Scalar{0.5} ? classes_(1) : classes_(0);
        }
        return labels;
    }

    [[nodiscard]] VectorType decision_function(const Eigen::Ref<const MatrixType>& input) const {
        this->check_is_fitted();
        this->validate_feature_count(input);
        const MatrixType k_trans = kernel_matrix(input, X_train_, /*same_matrix=*/false);
        return k_trans * dual_coef_;
    }

    [[nodiscard]] MatrixType predict_proba(const Eigen::Ref<const MatrixType>& input) const {
        this->check_is_fitted();
        this->validate_feature_count(input);
        const MatrixType k_trans = kernel_matrix(input, X_train_, /*same_matrix=*/false);
        const VectorType latent = k_trans * dual_coef_;
        const MatrixType weighted = w_sqrt_.asDiagonal() * k_trans.transpose();
        const MatrixType v = L_.template triangularView<Eigen::Lower>().solve(weighted);
        const MatrixType k_self = kernel_matrix(input, input, /*same_matrix=*/true);

        MatrixType probabilities(input.rows(), 2);
        for (IndexType row = 0; row < input.rows(); ++row) {
            const Scalar variance = std::max(
                Scalar{0}, k_self(row, row) - v.col(row).squaredNorm());
            const Scalar scale = std::sqrt(
                Scalar{1} + gaussian_process::detail::pi<Scalar>() * variance / Scalar{8});
            const Scalar positive = sigmoid(latent(row) / scale);
            probabilities(row, 0) = Scalar{1} - positive;
            probabilities(row, 1) = positive;
        }
        return probabilities;
    }

private:
    void validate_parameters() const {
        gaussian_process::detail::validate_kernel_parameters(
            kernel_,
            alpha_,
            length_scale_,
            constant_value_,
            noise_level_,
            nu_,
            rational_quadratic_alpha_,
            periodicity_,
            sigma_0_,
            "GaussianProcessClassifier");
        if (max_iter_ <= 0) {
            throw std::invalid_argument(
                "GaussianProcessClassifier: max_iter must be positive.");
        }
        if (tol_ < Scalar{0}) {
            throw std::invalid_argument(
                "GaussianProcessClassifier: tol must be non-negative.");
        }
    }

    void encode_classes(const Eigen::Ref<const LabelType>& target) {
        std::vector<int> unique;
        unique.reserve(static_cast<std::size_t>(target.size()));
        for (IndexType row = 0; row < target.size(); ++row) {
            unique.push_back(target(row));
        }
        std::sort(unique.begin(), unique.end());
        unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
        if (unique.size() != 2) {
            throw std::invalid_argument(
                "GaussianProcessClassifier: exactly two classes are supported.");
        }
        classes_.resize(2);
        classes_(0) = unique[0];
        classes_(1) = unique[1];
    }

    [[nodiscard]] VectorType laplace_step(const VectorType& f,
                                          const VectorType& y01,
                                          VectorType& alpha_out,
                                          VectorType& sqrt_w_out,
                                          MatrixType& l_out) const {
        const VectorType probabilities = f.unaryExpr(
            [](Scalar value) { return sigmoid(value); });
        VectorType w = (probabilities.array() * (Scalar{1} - probabilities.array())).matrix();
        w = w.array().max(Scalar{1e-12}).matrix();
        sqrt_w_out = w.array().sqrt().matrix();

        MatrixType b_matrix = MatrixType::Identity(f.size(), f.size());
        b_matrix.noalias() += sqrt_w_out.asDiagonal() * kernel_train_ * sqrt_w_out.asDiagonal();
        Eigen::LLT<MatrixType> llt(b_matrix);
        if (llt.info() != Eigen::Success) {
            throw std::runtime_error("GaussianProcessClassifier: Laplace factorization failed.");
        }
        l_out = llt.matrixL();

        const VectorType b = (w.array() * f.array() + y01.array() - probabilities.array()).matrix();
        const VectorType kb = kernel_train_ * b;
        const VectorType weighted_kb = sqrt_w_out.array() * kb.array();
        const VectorType solved = l_out.template triangularView<Eigen::Lower>().solve(weighted_kb);
        const VectorType solved_t = l_out.transpose()
                                    .template triangularView<Eigen::Upper>()
                                    .solve(solved);
        alpha_out = b - (sqrt_w_out.array() * solved_t.array()).matrix();
        return kernel_train_ * alpha_out;
    }

    [[nodiscard]] MatrixType kernel_matrix(const Eigen::Ref<const MatrixType>& A,
                                           const Eigen::Ref<const MatrixType>& B,
                                           bool same_matrix) const {
        return gaussian_process::detail::kernel_matrix(
            kernel_,
            A,
            B,
            same_matrix,
            length_scale_,
            constant_value_,
            noise_level_,
            nu_,
            rational_quadratic_alpha_,
            periodicity_,
            sigma_0_);
    }

    [[nodiscard]] static Scalar sigmoid(Scalar value) {
        if (value >= Scalar{0}) {
            const Scalar z = std::exp(-value);
            return Scalar{1} / (Scalar{1} + z);
        }
        const Scalar z = std::exp(value);
        return z / (Scalar{1} + z);
    }

    Kernel kernel_;
    Scalar alpha_;
    Scalar length_scale_;
    Scalar constant_value_;
    Scalar noise_level_;
    Scalar nu_;
    Scalar rational_quadratic_alpha_;
    Scalar periodicity_;
    Scalar sigma_0_;
    int max_iter_;
    Scalar tol_;

    LabelType classes_;
    MatrixType X_train_;
    MatrixType kernel_train_;
    MatrixType L_;
    VectorType w_sqrt_;
    VectorType dual_coef_;
    VectorType latent_mean_;
    int n_iter_ = 0;
};

/// @}

}  // namespace Skigen

#endif  // SKIGEN_GAUSSIAN_PROCESS_CLASSIFIER_H
