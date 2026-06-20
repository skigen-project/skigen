// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

#ifndef SKIGEN_DISCRIMINANT_ANALYSIS_H
#define SKIGEN_DISCRIMINANT_ANALYSIS_H

#include "../Core/Base.h"
#include "../Core/Validation.h"

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace Skigen {

/// @defgroup Algo_DiscriminantAnalysis Discriminant Analysis
/// @ingroup DiscriminantAnalysis
/// @brief Linear and quadratic discriminant analysis classifiers.
/// @{

namespace internal {

template <typename Scalar>
[[nodiscard]] Eigen::VectorXi sorted_unique_classes(const Eigen::VectorXi& y) {
    std::vector<int> values;
    values.reserve(static_cast<std::size_t>(y.size()));
    for (Eigen::Index i = 0; i < y.size(); ++i) values.push_back(y(i));
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    if (values.size() < 2) {
        throw std::invalid_argument("DiscriminantAnalysis: at least two classes are required.");
    }
    Eigen::VectorXi classes(static_cast<Eigen::Index>(values.size()));
    for (Eigen::Index i = 0; i < classes.size(); ++i) {
        classes(i) = values[static_cast<std::size_t>(i)];
    }
    return classes;
}

template <typename Scalar>
[[nodiscard]] Scalar logsumexp_row(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& scores,
                                   Eigen::Index row) {
    const Scalar max_score = scores.row(row).maxCoeff();
    Scalar sum = Scalar{0};
    for (Eigen::Index col = 0; col < scores.cols(); ++col) {
        sum += std::exp(scores(row, col) - max_score);
    }
    return max_score + std::log(sum);
}

}  // namespace internal

/// @brief Linear Discriminant Analysis classifier.
///
/// Fits class means, inferred class priors, and one pooled covariance matrix.
/// Prediction uses the standard linear discriminant score
/// @f$x^T\Sigma^{-1}\mu_k - \frac12\mu_k^T\Sigma^{-1}\mu_k + \log \pi_k@f$.
///
/// Mirrors the dense core of
/// [sklearn.discriminant_analysis.LinearDiscriminantAnalysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html).
///
/// ### Examples
///
/// @snippet discriminant_analysis.cpp example_lda
template <typename Scalar = double>
class LinearDiscriminantAnalysis
    : public Classifier<LinearDiscriminantAnalysis<Scalar>, Scalar> {
public:
    using Base = Classifier<LinearDiscriminantAnalysis<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::IndexType;
    using typename Base::LabelType;

    explicit LinearDiscriminantAnalysis(Scalar reg_param = Scalar{1e-9})
        : reg_param_(reg_param) {}

    [[nodiscard]] Scalar reg_param() const noexcept { return reg_param_; }

    [[nodiscard]] const Eigen::VectorXi& classes() const {
        this->check_is_fitted();
        return classes_;
    }

    [[nodiscard]] const VectorType& priors() const {
        this->check_is_fitted();
        return priors_;
    }

    [[nodiscard]] const MatrixType& means() const {
        this->check_is_fitted();
        return means_;
    }

    [[nodiscard]] const MatrixType& covariance() const {
        this->check_is_fitted();
        return covariance_;
    }

    [[nodiscard]] const MatrixType& coef() const {
        this->check_is_fitted();
        return coef_;
    }

    [[nodiscard]] const VectorType& intercept() const {
        this->check_is_fitted();
        return intercept_;
    }

    SKIGEN_PARAMS((reg_param, reg_param_, double))

    LinearDiscriminantAnalysis& fit_impl(const Eigen::Ref<const MatrixType>& X,
                                         const Eigen::Ref<const LabelType>& y) {
        internal::check_non_empty(X);
        internal::check_consistent_length(X, y);
        if (reg_param_ < Scalar{0}) {
            throw std::invalid_argument("LinearDiscriminantAnalysis: reg_param must be non-negative.");
        }

        classes_ = internal::sorted_unique_classes<Scalar>(y);
        const IndexType n_classes = classes_.size();
        const IndexType n_features = X.cols();
        this->n_features_in_ = n_features;

        means_ = MatrixType::Zero(n_classes, n_features);
        priors_ = VectorType::Zero(n_classes);
        covariance_ = MatrixType::Zero(n_features, n_features);

        for (IndexType cls_idx = 0; cls_idx < n_classes; ++cls_idx) {
            const int cls = classes_(cls_idx);
            IndexType count = 0;
            for (IndexType row = 0; row < y.size(); ++row) {
                if (y(row) == cls) {
                    means_.row(cls_idx) += X.row(row);
                    ++count;
                }
            }
            if (count == 0) {
                throw std::invalid_argument("LinearDiscriminantAnalysis: class with no samples.");
            }
            means_.row(cls_idx) /= static_cast<Scalar>(count);
            priors_(cls_idx) = static_cast<Scalar>(count) / static_cast<Scalar>(X.rows());

            for (IndexType row = 0; row < y.size(); ++row) {
                if (y(row) != cls) continue;
                const auto centered = (X.row(row) - means_.row(cls_idx)).eval();
                covariance_.noalias() += centered.transpose() * centered;
            }
        }

        const Scalar denom = std::max<Scalar>(Scalar{1}, static_cast<Scalar>(X.rows() - n_classes));
        covariance_ /= denom;
        covariance_.diagonal().array() += reg_param_;

        Eigen::LDLT<MatrixType> ldlt(covariance_);
        if (ldlt.info() != Eigen::Success) {
            throw std::runtime_error("LinearDiscriminantAnalysis: covariance factorization failed.");
        }
        coef_ = ldlt.solve(means_.transpose()).transpose();
        intercept_.resize(n_classes);
        for (IndexType cls_idx = 0; cls_idx < n_classes; ++cls_idx) {
            intercept_(cls_idx) =
                -Scalar{0.5} * means_.row(cls_idx).dot(coef_.row(cls_idx)) +
                std::log(priors_(cls_idx));
        }

        this->fitted_ = true;
        return *this;
    }

    [[nodiscard]] MatrixType decision_function(const Eigen::Ref<const MatrixType>& X) const {
        this->check_is_fitted();
        this->validate_feature_count(X);
        MatrixType scores = X * coef_.transpose();
        scores.rowwise() += intercept_.transpose();
        return scores;
    }

    [[nodiscard]] MatrixType predict_proba(const Eigen::Ref<const MatrixType>& X) const {
        MatrixType scores = decision_function(X);
        MatrixType probabilities(scores.rows(), scores.cols());
        for (IndexType row = 0; row < scores.rows(); ++row) {
            const Scalar norm = internal::logsumexp_row<Scalar>(scores, row);
            for (IndexType col = 0; col < scores.cols(); ++col) {
                probabilities(row, col) = std::exp(scores(row, col) - norm);
            }
        }
        return probabilities;
    }

    [[nodiscard]] MatrixType predict_log_proba(const Eigen::Ref<const MatrixType>& X) const {
        MatrixType scores = decision_function(X);
        for (IndexType row = 0; row < scores.rows(); ++row) {
            const Scalar norm = internal::logsumexp_row<Scalar>(scores, row);
            scores.row(row).array() -= norm;
        }
        return scores;
    }

    [[nodiscard]] LabelType predict_impl(const Eigen::Ref<const MatrixType>& X) const {
        const MatrixType scores = decision_function(X);
        LabelType out(X.rows());
        for (IndexType row = 0; row < scores.rows(); ++row) {
            Eigen::Index best = 0;
            scores.row(row).maxCoeff(&best);
            out(row) = classes_(best);
        }
        return out;
    }

private:
    Scalar reg_param_;
    Eigen::VectorXi classes_;
    VectorType priors_;
    MatrixType means_;
    MatrixType covariance_;
    MatrixType coef_;
    VectorType intercept_;
};

/// @brief Quadratic Discriminant Analysis classifier.
///
/// Fits one covariance model per class and predicts with the quadratic
/// Gaussian discriminant score.
///
/// Mirrors the dense core of
/// [sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html).
///
/// ### Examples
///
/// @snippet discriminant_analysis.cpp example_qda
template <typename Scalar = double>
class QuadraticDiscriminantAnalysis
    : public Classifier<QuadraticDiscriminantAnalysis<Scalar>, Scalar> {
public:
    using Base = Classifier<QuadraticDiscriminantAnalysis<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::IndexType;
    using typename Base::LabelType;

    explicit QuadraticDiscriminantAnalysis(Scalar reg_param = Scalar{0})
        : reg_param_(reg_param) {}

    [[nodiscard]] Scalar reg_param() const noexcept { return reg_param_; }

    [[nodiscard]] const Eigen::VectorXi& classes() const {
        this->check_is_fitted();
        return classes_;
    }

    [[nodiscard]] const VectorType& priors() const {
        this->check_is_fitted();
        return priors_;
    }

    [[nodiscard]] const MatrixType& means() const {
        this->check_is_fitted();
        return means_;
    }

    [[nodiscard]] const std::vector<MatrixType>& covariance() const {
        this->check_is_fitted();
        return covariance_;
    }

    SKIGEN_PARAMS((reg_param, reg_param_, double))

    QuadraticDiscriminantAnalysis& fit_impl(const Eigen::Ref<const MatrixType>& X,
                                            const Eigen::Ref<const LabelType>& y) {
        internal::check_non_empty(X);
        internal::check_consistent_length(X, y);
        if (reg_param_ < Scalar{0} || reg_param_ > Scalar{1}) {
            throw std::invalid_argument("QuadraticDiscriminantAnalysis: reg_param must be in [0, 1].");
        }

        classes_ = internal::sorted_unique_classes<Scalar>(y);
        const IndexType n_classes = classes_.size();
        const IndexType n_features = X.cols();
        this->n_features_in_ = n_features;
        means_ = MatrixType::Zero(n_classes, n_features);
        priors_ = VectorType::Zero(n_classes);
        covariance_.clear();
        covariance_.reserve(static_cast<std::size_t>(n_classes));
        inv_covariance_.clear();
        inv_covariance_.reserve(static_cast<std::size_t>(n_classes));
        log_det_.resize(n_classes);

        for (IndexType cls_idx = 0; cls_idx < n_classes; ++cls_idx) {
            const int cls = classes_(cls_idx);
            IndexType count = 0;
            for (IndexType row = 0; row < y.size(); ++row) {
                if (y(row) == cls) {
                    means_.row(cls_idx) += X.row(row);
                    ++count;
                }
            }
            if (count < 2) {
                throw std::invalid_argument(
                    "QuadraticDiscriminantAnalysis: each class needs at least two samples.");
            }
            means_.row(cls_idx) /= static_cast<Scalar>(count);
            priors_(cls_idx) = static_cast<Scalar>(count) / static_cast<Scalar>(X.rows());

            MatrixType cov = MatrixType::Zero(n_features, n_features);
            for (IndexType row = 0; row < y.size(); ++row) {
                if (y(row) != cls) continue;
                const auto centered = (X.row(row) - means_.row(cls_idx)).eval();
                cov.noalias() += centered.transpose() * centered;
            }
            cov /= static_cast<Scalar>(count - 1);
            cov = (Scalar{1} - reg_param_) * cov +
                  reg_param_ * MatrixType::Identity(n_features, n_features);

            Eigen::SelfAdjointEigenSolver<MatrixType> eig(cov);
            if (eig.info() != Eigen::Success) {
                throw std::runtime_error("QuadraticDiscriminantAnalysis: covariance eigensolve failed.");
            }
            const VectorType evals = eig.eigenvalues();
            const Scalar min_eval = evals.minCoeff();
            if (min_eval <= std::numeric_limits<Scalar>::epsilon()) {
                throw std::runtime_error(
                    "QuadraticDiscriminantAnalysis: covariance is rank deficient; increase reg_param.");
            }

            MatrixType inv = eig.eigenvectors() * evals.cwiseInverse().asDiagonal() * eig.eigenvectors().transpose();
            inv_covariance_.push_back(inv);
            covariance_.push_back(cov);
            log_det_(cls_idx) = evals.array().log().sum();
        }

        this->fitted_ = true;
        return *this;
    }

    [[nodiscard]] MatrixType decision_function(const Eigen::Ref<const MatrixType>& X) const {
        this->check_is_fitted();
        this->validate_feature_count(X);
        MatrixType scores(X.rows(), classes_.size());
        for (IndexType row = 0; row < X.rows(); ++row) {
            for (IndexType cls_idx = 0; cls_idx < classes_.size(); ++cls_idx) {
                const auto centered = (X.row(row) - means_.row(cls_idx)).transpose().eval();
                const Scalar mahalanobis = centered.dot(inv_covariance_[static_cast<std::size_t>(cls_idx)] * centered);
                scores(row, cls_idx) = -Scalar{0.5} * (mahalanobis + log_det_(cls_idx)) +
                                       std::log(priors_(cls_idx));
            }
        }
        return scores;
    }

    [[nodiscard]] MatrixType predict_proba(const Eigen::Ref<const MatrixType>& X) const {
        MatrixType scores = decision_function(X);
        MatrixType probabilities(scores.rows(), scores.cols());
        for (IndexType row = 0; row < scores.rows(); ++row) {
            const Scalar norm = internal::logsumexp_row<Scalar>(scores, row);
            for (IndexType col = 0; col < scores.cols(); ++col) {
                probabilities(row, col) = std::exp(scores(row, col) - norm);
            }
        }
        return probabilities;
    }

    [[nodiscard]] MatrixType predict_log_proba(const Eigen::Ref<const MatrixType>& X) const {
        MatrixType scores = decision_function(X);
        for (IndexType row = 0; row < scores.rows(); ++row) {
            const Scalar norm = internal::logsumexp_row<Scalar>(scores, row);
            scores.row(row).array() -= norm;
        }
        return scores;
    }

    [[nodiscard]] LabelType predict_impl(const Eigen::Ref<const MatrixType>& X) const {
        const MatrixType scores = decision_function(X);
        LabelType out(X.rows());
        for (IndexType row = 0; row < scores.rows(); ++row) {
            Eigen::Index best = 0;
            scores.row(row).maxCoeff(&best);
            out(row) = classes_(best);
        }
        return out;
    }

private:
    Scalar reg_param_;
    Eigen::VectorXi classes_;
    VectorType priors_;
    MatrixType means_;
    std::vector<MatrixType> covariance_;
    std::vector<MatrixType> inv_covariance_;
    VectorType log_det_;
};

/// @}

}  // namespace Skigen

#endif  // SKIGEN_DISCRIMINANT_ANALYSIS_H
