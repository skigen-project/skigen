// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_NAIVE_BAYES_GAUSSIAN_NB_H
#define SKIGEN_NAIVE_BAYES_GAUSSIAN_NB_H

#include "../Core/Base.h"
#include "../Core/Validation.h"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace Skigen {

/// @defgroup Algo_GaussianNB GaussianNB
/// @ingroup NaiveBayes
/// @brief Gaussian Naive Bayes classifier.
/// @{

/// @brief Gaussian Naive Bayes classifier.
///
/// Can perform online updates to model parameters via `partial_fit`.
/// The likelihood of the features is assumed to be Gaussian:
///
/// @f[
///   P(x_i | y) = \frac{1}{\sqrt{2\pi\sigma_y^2}}
///                 \exp\!\left(-\frac{(x_i-\mu_y)^2}{2\sigma_y^2}\right)
/// @f]
///
/// Mirrors
/// [sklearn.naive_bayes.GaussianNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html).
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `priors` | `VectorType` | empty | Prior probabilities of the classes. If empty, the priors are adjusted according to the data. |
/// | `var_smoothing` | `Scalar` | `1e-9` | Portion of the largest variance of all features that is added to variances for calculation stability. |
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type | Description |
/// |----------|------|-------------|
/// | `class_count()` | `Eigen::VectorXi` | Number of training samples observed in each class. |
/// | `class_prior()` | `VectorType` | Probability of each class. |
/// | `classes()` | `Eigen::VectorXi` | Class labels known to the classifier. |
/// | `epsilon()` | `Scalar` | Absolute additive value to variances. |
/// | `theta()` | `MatrixType` | Mean of each feature per class (n_classes × n_features). |
/// | `var()` | `MatrixType` | Variance of each feature per class (n_classes × n_features). |
///
/// ### See also
///
/// - Skigen::MultinomialNB — Naive Bayes for multinomially distributed data.
/// - Skigen::BernoulliNB — Naive Bayes for multivariate Bernoulli models.
///
/// ### Notes
///
/// The classifier supports online updates via `partial_fit` using the
/// numerically stable Welford / Chan parallel update so that batched
/// fitting yields bit-equivalent results to a single `fit` call on the
/// concatenated data.
///
/// ### Limitations relative to scikit-learn `sample_weight` in `fit` /
///   `partial_fit` is not honoured. Sklearn's `feature_names_in_` is
///   not exposed.
template <typename Scalar = double>
class GaussianNB : public Classifier<GaussianNB<Scalar>, Scalar> {
public:
    using Base = Classifier<GaussianNB<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;
    using typename Base::ScalarType;
    using typename Base::LabelType;

    /// @brief Construct a GaussianNB estimator.
    explicit GaussianNB(VectorType priors = VectorType(),
                        Scalar var_smoothing = Scalar{1e-9})
        : priors_(std::move(priors)), var_smoothing_(var_smoothing) {}

    // -- Accessors ----------------------------------------------------------

    [[nodiscard]] const Eigen::VectorXi& class_count() const {
        this->check_is_fitted(); return class_count_;
    }
    [[nodiscard]] const VectorType& class_prior() const {
        this->check_is_fitted(); return class_prior_;
    }
    [[nodiscard]] const Eigen::VectorXi& classes() const {
        this->check_is_fitted(); return classes_;
    }
    [[nodiscard]] Scalar epsilon() const {
        this->check_is_fitted(); return epsilon_;
    }
    [[nodiscard]] const MatrixType& theta() const {
        this->check_is_fitted(); return theta_;
    }
    [[nodiscard]] const MatrixType& var() const {
        this->check_is_fitted(); return var_;
    }

    SKIGEN_PARAMS((var_smoothing, var_smoothing_, double))

    // -- Implementation -----------------------------------------------------

    /// @brief Fit the model from scratch using a single training batch.
    GaussianNB& fit_impl(const Eigen::Ref<const MatrixType>& X,
                        const Eigen::Ref<const Eigen::VectorXi>& y) {
        internal::check_non_empty(X);
        internal::check_consistent_length(X, y);

        // Discover unique classes (sorted ascending)
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

        theta_ = MatrixType::Zero(n_classes, n_features);
        var_ = MatrixType::Zero(n_classes, n_features);
        class_count_ = Eigen::VectorXi::Zero(n_classes);

        // Compute epsilon = var_smoothing * max(variance over all features)
        // sklearn uses unbiased? Actually uses biased variance with N (population).
        // Var of full X column-wise (population variance):
        RowVectorType col_mean = X.colwise().mean();
        Scalar max_var = Scalar{0};
        for (Eigen::Index j = 0; j < n_features; ++j) {
            Scalar s = Scalar{0};
            for (Eigen::Index i = 0; i < X.rows(); ++i) {
                Scalar d = X(i, j) - col_mean(j);
                s += d * d;
            }
            s /= static_cast<Scalar>(X.rows());
            if (s > max_var) max_var = s;
        }
        epsilon_ = var_smoothing_ * max_var;

        // Per-class statistics (raw, no epsilon)
        for (Eigen::Index c = 0; c < n_classes; ++c) {
            int cls = classes_(c);
            int count = 0;
            for (Eigen::Index i = 0; i < y.size(); ++i) {
                if (y(i) == cls) ++count;
            }
            if (count == 0) {
                throw std::invalid_argument(
                    "Class with no samples encountered.");
            }
            class_count_(c) = count;

            // Compute mean
            RowVectorType mean = RowVectorType::Zero(n_features);
            for (Eigen::Index i = 0; i < y.size(); ++i) {
                if (y(i) == cls) mean += X.row(i);
            }
            mean /= static_cast<Scalar>(count);
            theta_.row(c) = mean;

            // Compute biased variance (population)
            RowVectorType var_row = RowVectorType::Zero(n_features);
            for (Eigen::Index i = 0; i < y.size(); ++i) {
                if (y(i) == cls) {
                    RowVectorType d = X.row(i) - mean;
                    var_row.array() += d.array().square();
                }
            }
            var_row /= static_cast<Scalar>(count);
            var_.row(c) = var_row;
        }

        // Compute class priors
        compute_class_prior();

        // Add epsilon to variances
        var_.array() += epsilon_;

        this->fitted_ = true;
        return *this;
    }

    /// @brief Predict labels for samples in X.
    [[nodiscard]] Eigen::VectorXi predict_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        MatrixType log_p = joint_log_likelihood(X);
        Eigen::VectorXi out(X.rows());
        for (Eigen::Index i = 0; i < X.rows(); ++i) {
            Eigen::Index idx;
            log_p.row(i).maxCoeff(&idx);
            out(i) = classes_(idx);
        }
        return out;
    }

    /// @brief Class probability estimates (n_samples × n_classes).
    [[nodiscard]] MatrixType predict_proba(
        const Eigen::Ref<const MatrixType>& X) const {
        this->check_is_fitted();
        this->validate_feature_count(X);
        MatrixType log_p = joint_log_likelihood(X);
        // normalize via log-sum-exp
        MatrixType proba(log_p.rows(), log_p.cols());
        for (Eigen::Index i = 0; i < log_p.rows(); ++i) {
            Scalar m = log_p.row(i).maxCoeff();
            Scalar s = Scalar{0};
            for (Eigen::Index c = 0; c < log_p.cols(); ++c) {
                s += std::exp(log_p(i, c) - m);
            }
            Scalar log_norm = m + std::log(s);
            for (Eigen::Index c = 0; c < log_p.cols(); ++c) {
                proba(i, c) = std::exp(log_p(i, c) - log_norm);
            }
        }
        return proba;
    }

    /// @brief Log of class probability estimates.
    [[nodiscard]] MatrixType predict_log_proba(
        const Eigen::Ref<const MatrixType>& X) const {
        this->check_is_fitted();
        this->validate_feature_count(X);
        MatrixType log_p = joint_log_likelihood(X);
        MatrixType log_proba(log_p.rows(), log_p.cols());
        for (Eigen::Index i = 0; i < log_p.rows(); ++i) {
            Scalar m = log_p.row(i).maxCoeff();
            Scalar s = Scalar{0};
            for (Eigen::Index c = 0; c < log_p.cols(); ++c) {
                s += std::exp(log_p(i, c) - m);
            }
            Scalar log_norm = m + std::log(s);
            for (Eigen::Index c = 0; c < log_p.cols(); ++c) {
                log_proba(i, c) = log_p(i, c) - log_norm;
            }
        }
        return log_proba;
    }

    /// @brief Incremental fit on a batch of samples.
    ///
    /// On the first call, `classes` must be provided (the full set of
    /// expected classes). On subsequent calls, an empty `classes` vector
    /// indicates re-use of the previously discovered classes.
    GaussianNB& partial_fit(const Eigen::Ref<const MatrixType>& X,
                            const Eigen::Ref<const Eigen::VectorXi>& y,
                            const Eigen::Ref<const Eigen::VectorXi>& classes) {
        internal::check_non_empty(X);
        internal::check_consistent_length(X, y);

        if (!this->fitted_) {
            if (classes.size() == 0) {
                throw std::invalid_argument(
                    "classes must be provided on the first call to partial_fit.");
            }
            // Initialize from `classes`
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

            theta_ = MatrixType::Zero(n_classes, n_features);
            var_ = MatrixType::Zero(n_classes, n_features);
            class_count_ = Eigen::VectorXi::Zero(n_classes);
            epsilon_ = Scalar{0};
        } else {
            if (X.cols() != this->n_features_in_) {
                throw std::invalid_argument(
                    "Number of features mismatch in partial_fit.");
            }
        }

        // Compute global epsilon based on this batch's variance (sklearn does
        // this every partial_fit call from the new batch only).
        RowVectorType col_mean = X.colwise().mean();
        Scalar max_var = Scalar{0};
        for (Eigen::Index j = 0; j < X.cols(); ++j) {
            Scalar s = Scalar{0};
            for (Eigen::Index i = 0; i < X.rows(); ++i) {
                Scalar d = X(i, j) - col_mean(j);
                s += d * d;
            }
            s /= static_cast<Scalar>(X.rows());
            if (s > max_var) max_var = s;
        }
        // Match sklearn: subtract previous epsilon, set new based on current
        // batch's max variance, then add new epsilon at the end.
        // We'll undo previous epsilon, compute fresh, and re-add.
        if (this->fitted_) {
            var_.array() -= epsilon_;
        }
        epsilon_ = var_smoothing_ * max_var;

        // Welford / Chan parallel update for each class
        for (Eigen::Index c = 0; c < classes_.size(); ++c) {
            int cls = classes_(c);
            int batch_count = 0;
            for (Eigen::Index i = 0; i < y.size(); ++i) {
                if (y(i) == cls) ++batch_count;
            }
            if (batch_count == 0) continue;

            // Compute batch mean
            RowVectorType batch_mean = RowVectorType::Zero(X.cols());
            for (Eigen::Index i = 0; i < y.size(); ++i) {
                if (y(i) == cls) batch_mean += X.row(i);
            }
            batch_mean /= static_cast<Scalar>(batch_count);

            // Compute batch (biased) variance
            RowVectorType batch_var = RowVectorType::Zero(X.cols());
            for (Eigen::Index i = 0; i < y.size(); ++i) {
                if (y(i) == cls) {
                    RowVectorType d = X.row(i) - batch_mean;
                    batch_var.array() += d.array().square();
                }
            }
            batch_var /= static_cast<Scalar>(batch_count);

            const Scalar n_old = static_cast<Scalar>(class_count_(c));
            const Scalar n_new = static_cast<Scalar>(batch_count);
            const Scalar n_total = n_old + n_new;

            if (n_old == Scalar{0}) {
                theta_.row(c) = batch_mean;
                var_.row(c) = batch_var;
            } else {
                RowVectorType old_mean = theta_.row(c);
                RowVectorType old_var = var_.row(c);
                // Chan/Welford: combined mean and variance
                RowVectorType new_mean =
                    (n_old * old_mean + n_new * batch_mean) / n_total;
                // Combined variance: weighted average + correction term
                RowVectorType delta = batch_mean - old_mean;
                RowVectorType new_var =
                    (n_old * old_var + n_new * batch_var) / n_total;
                new_var.array() +=
                    (n_old * n_new / (n_total * n_total)) * delta.array().square();
                theta_.row(c) = new_mean;
                var_.row(c) = new_var;
            }
            class_count_(c) += batch_count;
        }

        // Recompute class prior
        compute_class_prior();

        // Add epsilon
        var_.array() += epsilon_;

        this->fitted_ = true;
        return *this;
    }

private:
    VectorType priors_;
    Scalar var_smoothing_;

    Eigen::VectorXi class_count_;
    VectorType class_prior_;
    Eigen::VectorXi classes_;
    Scalar epsilon_ = Scalar{0};
    MatrixType theta_;
    MatrixType var_;

    void compute_class_prior() {
        const Eigen::Index n_classes = classes_.size();
        if (priors_.size() > 0) {
            if (priors_.size() != n_classes) {
                throw std::invalid_argument(
                    "Number of priors must match number of classes.");
            }
            class_prior_ = priors_;
        } else {
            class_prior_ = VectorType(n_classes);
            int total = class_count_.sum();
            for (Eigen::Index c = 0; c < n_classes; ++c) {
                class_prior_(c) = static_cast<Scalar>(class_count_(c)) /
                                  static_cast<Scalar>(total);
            }
        }
    }

    [[nodiscard]] MatrixType joint_log_likelihood(
        const Eigen::Ref<const MatrixType>& X) const {
        const Eigen::Index n = X.rows();
        const Eigen::Index n_classes = classes_.size();
        MatrixType jll(n, n_classes);
        const Scalar log_2pi = std::log(Scalar{2} *
            static_cast<Scalar>(3.14159265358979323846L));

        for (Eigen::Index c = 0; c < n_classes; ++c) {
            Scalar log_prior = std::log(class_prior_(c));
            // log p(x|y) = -0.5 * sum( log(2*pi*var) + (x-mu)^2/var )
            RowVectorType var_c = var_.row(c);
            RowVectorType mu_c = theta_.row(c);
            Scalar n_ij_const = Scalar{0};
            for (Eigen::Index j = 0; j < var_c.size(); ++j) {
                n_ij_const += std::log(var_c(j));
            }
            n_ij_const = -Scalar{0.5} *
                (static_cast<Scalar>(var_c.size()) * log_2pi + n_ij_const);
            for (Eigen::Index i = 0; i < n; ++i) {
                Scalar s = Scalar{0};
                for (Eigen::Index j = 0; j < var_c.size(); ++j) {
                    Scalar d = X(i, j) - mu_c(j);
                    s += (d * d) / var_c(j);
                }
                jll(i, c) = log_prior + n_ij_const - Scalar{0.5} * s;
            }
        }
        return jll;
    }
};

/// @}

} // namespace Skigen

#endif // SKIGEN_NAIVE_BAYES_GAUSSIAN_NB_H
