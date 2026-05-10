// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_NAIVE_BAYES_MULTINOMIAL_NB_H
#define SKIGEN_NAIVE_BAYES_MULTINOMIAL_NB_H

#include "../Core/Base.h"
#include "../Core/Validation.h"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace Skigen {

/// @defgroup Algo_MultinomialNB MultinomialNB
/// @ingroup NaiveBayes
/// @brief Naive Bayes classifier for multinomial models.
/// @{

/// @brief Naive Bayes classifier for multinomial models.
///
/// The multinomial distribution normally requires integer feature counts
/// (e.g. word counts for text classification). However in practice
/// fractional counts (e.g. tf-idf) may also work.
///
/// Mirrors
/// [sklearn.naive_bayes.MultinomialNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html).
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `alpha` | `Scalar` | `1.0` | Additive (Laplace/Lidstone) smoothing parameter. |
/// | `force_alpha` | `bool` | `true` | If `false`, alpha values smaller than `1e-10` are clipped. |
/// | `fit_prior` | `bool` | `true` | Whether to learn class prior probabilities. |
/// | `class_prior` | `VectorType` | empty | Prior probabilities of the classes. |
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type | Description |
/// |----------|------|-------------|
/// | `class_count()` | `VectorType` | Number of samples observed per class. |
/// | `class_log_prior()` | `VectorType` | Smoothed log prior for each class. |
/// | `classes()` | `Eigen::VectorXi` | Class labels known to the classifier. |
/// | `feature_count()` | `MatrixType` | Number of samples encountered for each (class, feature). |
/// | `feature_log_prob()` | `MatrixType` | Empirical log probability of features given a class, P(x_i \| y). |
/// | `n_features_in()` | `Eigen::Index` | Number of features seen during fitting. |
///
/// ### See also
///
/// - Skigen::GaussianNB — Gaussian Naive Bayes for continuous data.
/// - Skigen::BernoulliNB — Multivariate Bernoulli Naive Bayes.
///
/// ### Notes
///
/// Implements Laplace/Lidstone smoothing:
/// @f$ \log P(x_j | y) = \log\!\frac{N_{yj} + \alpha}{N_y + \alpha n} @f$.
///
/// ### Limitations relative to scikit-learn
///
/// `sample_weight` not supported.
///   `feature_names_in_` not exposed.
template <typename Scalar = double>
class MultinomialNB : public Classifier<MultinomialNB<Scalar>, Scalar> {
public:
    using Base = Classifier<MultinomialNB<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;
    using typename Base::ScalarType;
    using typename Base::LabelType;

    explicit MultinomialNB(Scalar alpha = Scalar{1},
                           bool force_alpha = true,
                           bool fit_prior = true,
                           VectorType class_prior = VectorType())
        : alpha_(alpha), force_alpha_(force_alpha), fit_prior_(fit_prior),
          class_prior_param_(std::move(class_prior)) {}

    // -- Accessors ----------------------------------------------------------

    [[nodiscard]] const VectorType& class_count() const {
        this->check_is_fitted(); return class_count_;
    }
    [[nodiscard]] const VectorType& class_log_prior() const {
        this->check_is_fitted(); return class_log_prior_;
    }
    [[nodiscard]] const Eigen::VectorXi& classes() const {
        this->check_is_fitted(); return classes_;
    }
    [[nodiscard]] const MatrixType& feature_count() const {
        this->check_is_fitted(); return feature_count_;
    }
    [[nodiscard]] const MatrixType& feature_log_prob() const {
        this->check_is_fitted(); return feature_log_prob_;
    }

    // -- Implementation -----------------------------------------------------

    MultinomialNB& fit_impl(const Eigen::Ref<const MatrixType>& X,
                            const Eigen::Ref<const Eigen::VectorXi>& y) {
        internal::check_non_empty(X);
        internal::check_consistent_length(X, y);

        // Discover classes
        std::vector<int> uniq;
        uniq.reserve(static_cast<std::size_t>(y.size()));
        for (Eigen::Index i = 0; i < y.size(); ++i) uniq.push_back(y(i));
        std::sort(uniq.begin(), uniq.end());
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());

        classes_ = Eigen::VectorXi(static_cast<Eigen::Index>(uniq.size()));
        for (std::size_t i = 0; i < uniq.size(); ++i)
            classes_(static_cast<Eigen::Index>(i)) = uniq[i];

        const Eigen::Index n_classes = classes_.size();
        const Eigen::Index n_features = X.cols();
        this->n_features_in_ = n_features;

        feature_count_ = MatrixType::Zero(n_classes, n_features);
        class_count_ = VectorType::Zero(n_classes);

        accumulate_counts(X, y);
        update_log_priors();
        update_feature_log_prob();

        this->fitted_ = true;
        return *this;
    }

    [[nodiscard]] Eigen::VectorXi predict_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        MatrixType jll = joint_log_likelihood(X);
        Eigen::VectorXi out(X.rows());
        for (Eigen::Index i = 0; i < X.rows(); ++i) {
            Eigen::Index idx;
            jll.row(i).maxCoeff(&idx);
            out(i) = classes_(idx);
        }
        return out;
    }

    [[nodiscard]] MatrixType predict_proba(
        const Eigen::Ref<const MatrixType>& X) const {
        this->check_is_fitted();
        this->validate_feature_count(X);
        return normalize_log_jll(joint_log_likelihood(X), /*as_log=*/false);
    }

    [[nodiscard]] MatrixType predict_log_proba(
        const Eigen::Ref<const MatrixType>& X) const {
        this->check_is_fitted();
        this->validate_feature_count(X);
        return normalize_log_jll(joint_log_likelihood(X), /*as_log=*/true);
    }

    MultinomialNB& partial_fit(const Eigen::Ref<const MatrixType>& X,
                               const Eigen::Ref<const Eigen::VectorXi>& y,
                               const Eigen::Ref<const Eigen::VectorXi>& classes) {
        internal::check_non_empty(X);
        internal::check_consistent_length(X, y);

        if (!this->fitted_) {
            if (classes.size() == 0) {
                throw std::invalid_argument(
                    "classes must be provided on the first call to partial_fit.");
            }
            std::vector<int> uniq;
            uniq.reserve(static_cast<std::size_t>(classes.size()));
            for (Eigen::Index i = 0; i < classes.size(); ++i)
                uniq.push_back(classes(i));
            std::sort(uniq.begin(), uniq.end());
            uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());

            classes_ = Eigen::VectorXi(static_cast<Eigen::Index>(uniq.size()));
            for (std::size_t i = 0; i < uniq.size(); ++i)
                classes_(static_cast<Eigen::Index>(i)) = uniq[i];

            const Eigen::Index n_classes = classes_.size();
            const Eigen::Index n_features = X.cols();
            this->n_features_in_ = n_features;
            feature_count_ = MatrixType::Zero(n_classes, n_features);
            class_count_ = VectorType::Zero(n_classes);
        } else {
            if (X.cols() != this->n_features_in_) {
                throw std::invalid_argument(
                    "Number of features mismatch in partial_fit.");
            }
        }

        accumulate_counts(X, y);
        update_log_priors();
        update_feature_log_prob();

        this->fitted_ = true;
        return *this;
    }

protected:
    Scalar alpha_;
    bool force_alpha_;
    bool fit_prior_;
    VectorType class_prior_param_;

    Eigen::VectorXi classes_;
    VectorType class_count_;
    VectorType class_log_prior_;
    MatrixType feature_count_;
    MatrixType feature_log_prob_;

    [[nodiscard]] Scalar effective_alpha() const {
        if (!force_alpha_ && alpha_ < Scalar{1e-10}) {
            return Scalar{1e-10};
        }
        return alpha_;
    }

    void accumulate_counts(const Eigen::Ref<const MatrixType>& X,
                           const Eigen::Ref<const Eigen::VectorXi>& y) {
        for (Eigen::Index c = 0; c < classes_.size(); ++c) {
            int cls = classes_(c);
            for (Eigen::Index i = 0; i < y.size(); ++i) {
                if (y(i) == cls) {
                    feature_count_.row(c) += X.row(i);
                    class_count_(c) += Scalar{1};
                }
            }
        }
    }

    void update_log_priors() {
        const Eigen::Index n_classes = classes_.size();
        class_log_prior_ = VectorType(n_classes);

        if (class_prior_param_.size() > 0) {
            if (class_prior_param_.size() != n_classes) {
                throw std::invalid_argument(
                    "Number of priors must match number of classes.");
            }
            for (Eigen::Index c = 0; c < n_classes; ++c) {
                class_log_prior_(c) = std::log(class_prior_param_(c));
            }
        } else if (fit_prior_) {
            Scalar total = class_count_.sum();
            for (Eigen::Index c = 0; c < n_classes; ++c) {
                class_log_prior_(c) =
                    std::log(class_count_(c) / total);
            }
        } else {
            Scalar lp = -std::log(static_cast<Scalar>(n_classes));
            for (Eigen::Index c = 0; c < n_classes; ++c) {
                class_log_prior_(c) = lp;
            }
        }
    }

    void update_feature_log_prob() {
        const Eigen::Index n_classes = classes_.size();
        const Eigen::Index n_features = feature_count_.cols();
        const Scalar a = effective_alpha();

        MatrixType smoothed_fc = feature_count_.array() + a;
        // Sum over features per class
        VectorType smoothed_cc(n_classes);
        for (Eigen::Index c = 0; c < n_classes; ++c) {
            smoothed_cc(c) = smoothed_fc.row(c).sum();
        }

        feature_log_prob_ = MatrixType(n_classes, n_features);
        for (Eigen::Index c = 0; c < n_classes; ++c) {
            Scalar log_denom = std::log(smoothed_cc(c));
            for (Eigen::Index j = 0; j < n_features; ++j) {
                feature_log_prob_(c, j) =
                    std::log(smoothed_fc(c, j)) - log_denom;
            }
        }
    }

    [[nodiscard]] virtual MatrixType joint_log_likelihood(
        const Eigen::Ref<const MatrixType>& X) const {
        // jll = X * feature_log_prob_^T + class_log_prior_
        MatrixType jll = X * feature_log_prob_.transpose();
        jll.rowwise() += class_log_prior_.transpose();
        return jll;
    }

    [[nodiscard]] MatrixType normalize_log_jll(MatrixType jll,
                                               bool as_log) const {
        for (Eigen::Index i = 0; i < jll.rows(); ++i) {
            Scalar m = jll.row(i).maxCoeff();
            Scalar s = Scalar{0};
            for (Eigen::Index c = 0; c < jll.cols(); ++c) {
                s += std::exp(jll(i, c) - m);
            }
            Scalar log_norm = m + std::log(s);
            for (Eigen::Index c = 0; c < jll.cols(); ++c) {
                if (as_log) {
                    jll(i, c) = jll(i, c) - log_norm;
                } else {
                    jll(i, c) = std::exp(jll(i, c) - log_norm);
                }
            }
        }
        return jll;
    }
};

/// @}

} // namespace Skigen

#endif // SKIGEN_NAIVE_BAYES_MULTINOMIAL_NB_H
