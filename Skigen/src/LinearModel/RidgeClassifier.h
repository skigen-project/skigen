// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

#ifndef SKIGEN_LINEAR_MODEL_RIDGE_CLASSIFIER_H
#define SKIGEN_LINEAR_MODEL_RIDGE_CLASSIFIER_H

#include "../Core/Base.h"
#include "../Core/Validation.h"

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace Skigen {

/// @defgroup Algo_RidgeClassifier RidgeClassifier
/// @ingroup LinearModels
/// @brief Ridge classifier.
/// @{

/// @brief Classifier using ridge regression on encoded class targets.
///
/// Binary classification fits one target column encoded as `-1` / `+1`.
/// Multiclass classification fits one-vs-rest encoded targets.
///
/// Mirrors the dense core of
/// [sklearn.linear_model.RidgeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html).
///
/// ### Examples
///
/// @snippet ridge_classifier.cpp example_ridge_classifier
template <typename Scalar = double>
class RidgeClassifier : public Classifier<RidgeClassifier<Scalar>, Scalar> {
public:
    using Base = Classifier<RidgeClassifier<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;
    using typename Base::LabelType;

    explicit RidgeClassifier(Scalar alpha = Scalar{1}, bool fit_intercept = true)
        : alpha_(alpha), fit_intercept_(fit_intercept) {}

    [[nodiscard]] Scalar alpha() const noexcept { return alpha_; }
    [[nodiscard]] bool fit_intercept() const noexcept { return fit_intercept_; }

    [[nodiscard]] const Eigen::VectorXi& classes() const {
        this->check_is_fitted();
        return classes_;
    }

    [[nodiscard]] const MatrixType& coef() const {
        this->check_is_fitted();
        return coef_;
    }

    [[nodiscard]] const VectorType& intercept() const {
        this->check_is_fitted();
        return intercept_;
    }

    SKIGEN_PARAMS(
        (alpha, alpha_, double),
        (fit_intercept, fit_intercept_, bool))

    RidgeClassifier& fit_impl(const Eigen::Ref<const MatrixType>& X,
                              const Eigen::Ref<const LabelType>& y) {
        internal::check_non_empty(X);
        internal::check_consistent_length(X, y);
        if (alpha_ < Scalar{0}) {
            throw std::invalid_argument("RidgeClassifier: alpha must be non-negative.");
        }

        classes_ = sorted_unique(y);
        if (classes_.size() < 2) {
            throw std::invalid_argument("RidgeClassifier: at least two classes are required.");
        }

        this->n_features_in_ = X.cols();
        const IndexType output_count = classes_.size() == 2 ? 1 : classes_.size();
        MatrixType Y(X.rows(), output_count);
        if (classes_.size() == 2) {
            for (IndexType row = 0; row < y.size(); ++row) {
                Y(row, 0) = y(row) == classes_(1) ? Scalar{1} : Scalar{-1};
            }
        } else {
            for (IndexType row = 0; row < y.size(); ++row) {
                for (IndexType cls = 0; cls < classes_.size(); ++cls) {
                    Y(row, cls) = y(row) == classes_(cls) ? Scalar{1} : Scalar{-1};
                }
            }
        }

        MatrixType Xc;
        MatrixType Yc;
        RowVectorType x_offset = RowVectorType::Zero(X.cols());
        RowVectorType y_offset = RowVectorType::Zero(output_count);
        if (fit_intercept_) {
            x_offset = X.colwise().mean();
            y_offset = Y.colwise().mean();
            Xc = X.rowwise() - x_offset;
            Yc = Y.rowwise() - y_offset;
        } else {
            Xc = X;
            Yc = Y;
        }

        MatrixType XtX = Xc.transpose() * Xc;
        XtX.diagonal().array() += alpha_;
        MatrixType XtY = Xc.transpose() * Yc;
        Eigen::LDLT<MatrixType> ldlt(XtX);
        MatrixType W = ldlt.solve(XtY);

        coef_ = W.transpose();
        intercept_.resize(output_count);
        if (fit_intercept_) {
            for (IndexType out = 0; out < output_count; ++out) {
                intercept_(out) = y_offset(out) - x_offset.dot(W.col(out));
            }
        } else {
            intercept_.setZero();
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

    [[nodiscard]] LabelType predict_impl(const Eigen::Ref<const MatrixType>& X) const {
        const MatrixType scores = decision_function(X);
        LabelType out(X.rows());
        if (classes_.size() == 2) {
            for (IndexType row = 0; row < X.rows(); ++row) {
                out(row) = scores(row, 0) > Scalar{0} ? classes_(1) : classes_(0);
            }
            return out;
        }
        for (IndexType row = 0; row < X.rows(); ++row) {
            Eigen::Index best = 0;
            scores.row(row).maxCoeff(&best);
            out(row) = classes_(best);
        }
        return out;
    }

private:
    [[nodiscard]] static Eigen::VectorXi sorted_unique(const LabelType& y) {
        std::vector<int> values;
        values.reserve(static_cast<std::size_t>(y.size()));
        for (IndexType i = 0; i < y.size(); ++i) values.push_back(y(i));
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end()), values.end());
        Eigen::VectorXi classes(static_cast<IndexType>(values.size()));
        for (IndexType i = 0; i < classes.size(); ++i) {
            classes(i) = values[static_cast<std::size_t>(i)];
        }
        return classes;
    }

    Scalar alpha_;
    bool fit_intercept_;
    Eigen::VectorXi classes_;
    MatrixType coef_;
    VectorType intercept_;
};

/// @}

}  // namespace Skigen

#endif  // SKIGEN_LINEAR_MODEL_RIDGE_CLASSIFIER_H
