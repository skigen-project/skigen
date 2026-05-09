// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_NAIVE_BAYES_BERNOULLI_NB_H
#define SKIGEN_NAIVE_BAYES_BERNOULLI_NB_H

#include "../Core/Base.h"
#include "../Core/Validation.h"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <vector>

namespace Skigen {

/// @defgroup Algo_BernoulliNB BernoulliNB
/// @ingroup NaiveBayes
/// @brief Naive Bayes classifier for multivariate Bernoulli models.
/// @{

/// @brief Naive Bayes classifier for multivariate Bernoulli models.
///
/// Like MultinomialNB, this classifier is suitable for discrete data.
/// The difference is that while MultinomialNB works with occurrence
/// counts, BernoulliNB is designed for binary/boolean features.
///
/// Mirrors
/// [sklearn.naive_bayes.BernoulliNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html).
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `alpha` | `Scalar` | `1.0` | Additive (Laplace/Lidstone) smoothing parameter. |
/// | `force_alpha` | `bool` | `true` | If `false`, alpha values smaller than `1e-10` are clipped. |
/// | `binarize` | `std::optional<Scalar>` | `0.0` | Threshold for binarizing input features. If `nullopt`, input is assumed already binary. |
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
/// | `feature_log_prob()` | `MatrixType` | Empirical log probability that a feature is 1 given a class. |
/// | `n_features_in()` | `Eigen::Index` | Number of features seen during fitting. |
///
/// ### See also
///
/// - Skigen::MultinomialNB — Naive Bayes for multinomial data.
/// - Skigen::GaussianNB — Gaussian Naive Bayes.
///
/// ### Notes
///
/// The decision rule for BernoulliNB is based on
/// @f$ P(x_i | y) = P(i|y)x_i + (1 - P(i|y))(1 - x_i) @f$.
///
/// @note **scikit-learn parity gap:** `sample_weight` not supported.
///   `feature_names_in_` not exposed.
template <typename Scalar = double>
class BernoulliNB : public Classifier<BernoulliNB<Scalar>, Scalar> {
public:
    using Base = Classifier<BernoulliNB<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;
    using typename Base::ScalarType;
    using typename Base::LabelType;

    explicit BernoulliNB(Scalar alpha = Scalar{1},
                         bool force_alpha = true,
                         std::optional<Scalar> binarize = Scalar{0},
                         bool fit_prior = true,
                         VectorType class_prior = VectorType())
        : alpha_(alpha), force_alpha_(force_alpha), binarize_(binarize),
          fit_prior_(fit_prior),
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

    BernoulliNB& fit_impl(const Eigen::Ref<const MatrixType>& X,
                          const Eigen::Ref<const Eigen::VectorXi>& y) {
        internal::check_non_empty(X);
        internal::check_consistent_length(X, y);

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

        MatrixType Xb = binarize_input(X);
        accumulate_counts(Xb, y);
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

    BernoulliNB& partial_fit(const Eigen::Ref<const MatrixType>& X,
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

        MatrixType Xb = binarize_input(X);
        accumulate_counts(Xb, y);
        update_log_priors();
        update_feature_log_prob();

        this->fitted_ = true;
        return *this;
    }

private:
    Scalar alpha_;
    bool force_alpha_;
    std::optional<Scalar> binarize_;
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

    [[nodiscard]] MatrixType binarize_input(
        const Eigen::Ref<const MatrixType>& X) const {
        if (!binarize_.has_value()) {
            return X;
        }
        Scalar t = *binarize_;
        MatrixType out(X.rows(), X.cols());
        for (Eigen::Index i = 0; i < X.rows(); ++i) {
            for (Eigen::Index j = 0; j < X.cols(); ++j) {
                out(i, j) = (X(i, j) > t) ? Scalar{1} : Scalar{0};
            }
        }
        return out;
    }

    void accumulate_counts(const MatrixType& Xb,
                           const Eigen::Ref<const Eigen::VectorXi>& y) {
        for (Eigen::Index c = 0; c < classes_.size(); ++c) {
            int cls = classes_(c);
            for (Eigen::Index i = 0; i < y.size(); ++i) {
                if (y(i) == cls) {
                    feature_count_.row(c) += Xb.row(i);
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

        // smoothed_fc = feature_count + alpha
        // smoothed_cc = class_count + 2*alpha (Bernoulli: 2 outcomes)
        feature_log_prob_ = MatrixType(n_classes, n_features);
        for (Eigen::Index c = 0; c < n_classes; ++c) {
            Scalar log_denom = std::log(class_count_(c) + Scalar{2} * a);
            for (Eigen::Index j = 0; j < n_features; ++j) {
                feature_log_prob_(c, j) =
                    std::log(feature_count_(c, j) + a) - log_denom;
            }
        }
    }

    [[nodiscard]] MatrixType joint_log_likelihood(
        const Eigen::Ref<const MatrixType>& X) const {
        MatrixType Xb = binarize_input(X);
        const Eigen::Index n_classes = classes_.size();
        const Eigen::Index n_features = feature_log_prob_.cols();

        // neg_prob(c,j) = log(1 - exp(feature_log_prob_(c,j)))
        MatrixType neg_prob(n_classes, n_features);
        for (Eigen::Index c = 0; c < n_classes; ++c) {
            for (Eigen::Index j = 0; j < n_features; ++j) {
                neg_prob(c, j) =
                    std::log(Scalar{1} -
                             std::exp(feature_log_prob_(c, j)));
            }
        }

        // jll = X * (flp - neg_prob)^T + sum(neg_prob, axis=1) + class_log_prior
        MatrixType diff = feature_log_prob_ - neg_prob;
        MatrixType jll = Xb * diff.transpose();

        VectorType neg_row_sum(n_classes);
        for (Eigen::Index c = 0; c < n_classes; ++c) {
            neg_row_sum(c) = neg_prob.row(c).sum();
        }

        for (Eigen::Index i = 0; i < jll.rows(); ++i) {
            for (Eigen::Index c = 0; c < n_classes; ++c) {
                jll(i, c) += class_log_prior_(c) + neg_row_sum(c);
            }
        }
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

#endif // SKIGEN_NAIVE_BAYES_BERNOULLI_NB_H
