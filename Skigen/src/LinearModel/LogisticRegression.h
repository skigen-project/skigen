// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_LINEAR_MODEL_LOGISTIC_REGRESSION_H
#define SKIGEN_LINEAR_MODEL_LOGISTIC_REGRESSION_H

#include "../Core/Base.h"
#include "../Core/Validation.h"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

namespace Skigen {

/// @defgroup Algo_LogisticRegression LogisticRegression
/// @ingroup LinearModels
/// @brief Logistic regression classifier (L2, IRLS solver).
/// @see https://skigen-project.github.io/docs/guide/logistic-regression for algorithm intuition.
/// @{

/// @brief Logistic Regression (aka logit, MaxEnt) classifier.
///
/// In the multiclass case, the training algorithm uses a one-vs-rest
/// (OvR) scheme. Each binary sub-problem is solved with an
/// Iteratively Reweighted Least Squares (IRLS / Newton) solver that
/// minimizes:
///
/// @f[
///   \min_w \;\frac{1}{n}\sum_{i=1}^n
///     \log\!\bigl(1 + e^{-y_i (X_i w + b)}\bigr)
///     + \frac{1}{2C}\|w\|_2^2
/// @f]
///
/// Mirrors
/// [sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `C` | `Scalar` | `1` | Inverse of regularization strength; must be a positive float. Smaller values specify stronger regularization. |
/// | `fit_intercept` | `bool` | `true` | Whether the intercept should be estimated. |
/// | `max_iter` | `int` | `100` | Maximum number of IRLS iterations. |
/// | `tol` | `Scalar` | `1e-4` | Tolerance for stopping criteria (max coordinate update). |
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type | Description |
/// |----------|------|-------------|
/// | `coef()` | `MatrixType` | Coefficient matrix of shape (n_classes, n_features) or (1, n_features) for binary. |
/// | `intercept()` | `VectorType` | Intercept (bias) vector of shape (n_classes,) or (1,). |
/// | `classes()` | `std::vector<int>` | Unique class labels sorted in ascending order. |
///
/// ### See also
///
/// - Skigen::SGDClassifier — Linear classifier trained via SGD (supports hinge and log loss).
///
/// ### Notes
///
/// The solver is a Newton/IRLS method with diagonal Hessian
/// approximation. It converges quickly for well-scaled data, but
/// `StandardScaler` is recommended for best performance.
///
/// ### Limitations relative to scikit-learn
///
/// The following scikit-learn constructor
///   parameters are not honoured: `penalty` (only L2 is implemented),
///   `dual`, `solver` (only IRLS), `multi_class` (only OvR), `class_weight`,
///   `verbose`, `warm_start`, `n_jobs`, `l1_ratio`.
///   The following sklearn fitted attributes are not exposed:
///   `n_iter_`, `n_features_in_`, `feature_names_in_`.
///   The sklearn methods `decision_function()` and `predict_log_proba()`
///   are not yet public.
///   `sample_weight` in `fit()` is not honoured.
///
/// ### Examples
///
/// @snippet logistic_regression.cpp example_logistic_regression
template <typename Scalar = double>
class LogisticRegression
    : public Classifier<LogisticRegression<Scalar>, Scalar> {
public:
    using Base = Classifier<LogisticRegression<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;
    using typename Base::ScalarType;
    using typename Base::LabelType;
    using IndexVector = Eigen::VectorXi;

    /// @brief Construct a LogisticRegression estimator.
    ///
    /// @param C Inverse of regularization strength (`Scalar`, default `1`).
    ///   Must be positive. Smaller values specify stronger regularization.
    /// @param fit_intercept Whether the intercept should be estimated (`bool`, default `true`).
    /// @param max_iter Maximum number of IRLS iterations (`int`, default `100`).
    /// @param tol Tolerance for stopping criteria (`Scalar`, default `1e-4`).
    explicit LogisticRegression(Scalar C = Scalar{1}, bool fit_intercept = true,
                                int max_iter = 100, Scalar tol = Scalar{1e-4})
        : C_(C), fit_intercept_(fit_intercept), max_iter_(max_iter), tol_(tol) {}

    /// @brief Coefficient matrix (n_classes × n_features or 1 × n_features for binary).
    [[nodiscard]] const MatrixType& coef() const {
        this->check_is_fitted();
        return coef_;
    }
    /// @brief Intercept (bias) vector of shape (n_classes,) or (1,).
    [[nodiscard]] const VectorType& intercept() const {
        this->check_is_fitted();
        return intercept_;
    }
    /// @brief Unique class labels sorted in ascending order.
    ///
    /// @return Read-only reference to the class label vector.
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] const std::vector<int>& classes() const {
        this->check_is_fitted();
        return classes_;
    }

    SKIGEN_PARAMS(
        (C,             C_,             double),
        (fit_intercept, fit_intercept_, bool),
        (max_iter,      max_iter_,      int),
        (tol,           tol_,           double))

    /// @brief Fit the model according to the given training data.
    ///
    /// Discovers unique classes in `y`, then solves binary logistic
    /// regression sub-problems (OvR for multiclass) via IRLS.
    ///
    /// @param X Training matrix of shape (n_samples, n_features).
    /// @param y Target vector of shape (n_samples,) with integer class labels.
    /// @return Reference to the fitted estimator (`*this`).
    /// @throws std::invalid_argument if fewer than 2 classes are found
    ///   or X and y have inconsistent lengths.
    ///
    /// ### Limitations relative to scikit-learn
///
/// `sample_weight`, `class_weight`
    ///   parameters are not honoured.
    LogisticRegression& fit_impl(const Eigen::Ref<const MatrixType>& X,
                            const Eigen::Ref<const IndexVector>& y) {
        internal::check_non_empty(X);
        internal::check_consistent_length(X, y);

        this->n_features_in_ = X.cols();

        // Discover classes
        std::map<int, int> class_map;
        for (Eigen::Index i = 0; i < y.size(); ++i) {
            class_map.emplace(y(i), 0);
        }
        classes_.clear();
        int idx = 0;
        for (auto& [cls, id] : class_map) {
            id = idx++;
            classes_.push_back(cls);
        }

        const int n_classes = static_cast<int>(classes_.size());

        if (n_classes < 2) {
            throw std::invalid_argument("Need at least 2 classes.");
        }

        if (n_classes == 2) {
            // Binary: single weight vector
            coef_.resize(1, X.cols());
            intercept_.resize(1);

            VectorType binary_y(y.size());
            for (Eigen::Index i = 0; i < y.size(); ++i) {
                binary_y(i) = (class_map[y(i)] == 1) ? Scalar{1} : Scalar{0};
            }

            RowVectorType w;
            Scalar b;
            fit_binary(X, binary_y, w, b);
            coef_.row(0) = w;
            intercept_(0) = b;
        } else {
            // One-vs-Rest
            coef_.resize(n_classes, X.cols());
            intercept_.resize(n_classes);

            for (int c = 0; c < n_classes; ++c) {
                VectorType binary_y(y.size());
                for (Eigen::Index i = 0; i < y.size(); ++i) {
                    binary_y(i) = (class_map[y(i)] == c) ? Scalar{1} : Scalar{0};
                }

                RowVectorType w;
                Scalar b;
                fit_binary(X, binary_y, w, b);
                coef_.row(c) = w;
                intercept_(c) = b;
            }
        }

        this->fitted_ = true;
        return *this;
    }

    /// @brief Predict class labels for samples in X.
    [[nodiscard]] IndexVector predict_impl(
        const Eigen::Ref<const MatrixType>& X) const {

        MatrixType scores = decision_function(X);
        IndexVector predictions(X.rows());

        if (classes_.size() == 2) {
            for (Eigen::Index i = 0; i < X.rows(); ++i) {
                predictions(i) = scores(i, 0) >= Scalar{0}
                    ? classes_[1] : classes_[0];
            }
        } else {
            for (Eigen::Index i = 0; i < X.rows(); ++i) {
                Eigen::Index max_idx;
                scores.row(i).maxCoeff(&max_idx);
                predictions(i) = classes_[static_cast<std::size_t>(max_idx)];
            }
        }

        return predictions;
    }

    /// @brief Probability estimates for each class.
    ///
    /// Returns a matrix of shape (n_samples, n_classes) where each
    /// row sums to 1. For binary, column 0 is the probability of
    /// class 0, column 1 of class 1. For multiclass, probabilities
    /// are normalized per-row.
    ///
    /// @param X Sample matrix of shape (n_samples, n_features).
    /// @return Probability matrix (n_samples, n_classes).
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] MatrixType predict_proba(
        const Eigen::Ref<const MatrixType>& X) const {
        this->check_is_fitted();
        this->validate_feature_count(X);

        MatrixType scores = decision_function(X);

        if (classes_.size() == 2) {
            MatrixType proba(X.rows(), 2);
            for (Eigen::Index i = 0; i < X.rows(); ++i) {
                Scalar p1 = sigmoid(scores(i, 0));
                proba(i, 0) = Scalar{1} - p1;
                proba(i, 1) = p1;
            }
            return proba;
        } else {
            // Softmax-like: sigmoid per class, then normalize
            MatrixType proba(X.rows(), static_cast<Eigen::Index>(classes_.size()));
            for (Eigen::Index i = 0; i < X.rows(); ++i) {
                Scalar sum{0};
                for (Eigen::Index c = 0; c < proba.cols(); ++c) {
                    proba(i, c) = sigmoid(scores(i, c));
                    sum += proba(i, c);
                }
                if (sum > Scalar{0}) proba.row(i) /= sum;
            }
            return proba;
        }
    }

private:
    Scalar C_;
    bool fit_intercept_;
    int max_iter_;
    Scalar tol_;

    MatrixType coef_;
    VectorType intercept_;
    std::vector<int> classes_;

    static Scalar sigmoid(Scalar x) {
        // Numerically stable sigmoid
        if (x >= Scalar{0}) {
            Scalar z = std::exp(-x);
            return Scalar{1} / (Scalar{1} + z);
        } else {
            Scalar z = std::exp(x);
            return z / (Scalar{1} + z);
        }
    }

    [[nodiscard]] MatrixType decision_function(
        const Eigen::Ref<const MatrixType>& X) const {
        // (n_samples x n_classifiers) = X * coef^T + intercept
        MatrixType scores = X * coef_.transpose();
        scores.rowwise() += intercept_.transpose();
        return scores;
    }

    // Binary logistic regression via L-BFGS-like gradient descent
    void fit_binary(const Eigen::Ref<const MatrixType>& X,
                    const VectorType& y,
                    RowVectorType& w_out, Scalar& b_out) const {
        const Eigen::Index n = X.rows();
        const Eigen::Index p = X.cols();
        const Scalar reg = Scalar{1} / C_;

        RowVectorType w = RowVectorType::Zero(p);
        Scalar b{0};

        for (int iter = 0; iter < max_iter_; ++iter) {
            // Compute predictions
            VectorType linear = (X * w.transpose()).array() + b;
            VectorType prob(n);
            for (Eigen::Index i = 0; i < n; ++i) {
                prob(i) = sigmoid(linear(i));
            }

            // Gradient
            VectorType diff = prob - y;
            RowVectorType grad_w = (diff.transpose() * X) / static_cast<Scalar>(n)
                                   + reg * w;
            Scalar grad_b = fit_intercept_
                ? diff.sum() / static_cast<Scalar>(n) : Scalar{0};

            // Hessian diagonal approximation (Newton step)
            VectorType s = prob.array() * (Scalar{1} - prob.array());
            s = s.array().max(Scalar{1e-12}); // numerical stability

            RowVectorType H_diag(p);
            for (Eigen::Index j = 0; j < p; ++j) {
                H_diag(j) = (X.col(j).array().square() * s.array()).sum()
                             / static_cast<Scalar>(n) + reg;
            }

            // Update
            Scalar max_change{0};
            for (Eigen::Index j = 0; j < p; ++j) {
                Scalar delta = grad_w(j) / H_diag(j);
                w(j) -= delta;
                max_change = std::max(max_change, std::abs(delta));
            }

            if (fit_intercept_) {
                Scalar H_b = s.sum() / static_cast<Scalar>(n);
                if (H_b > Scalar{1e-12}) {
                    Scalar delta_b = grad_b / H_b;
                    b -= delta_b;
                    max_change = std::max(max_change, std::abs(delta_b));
                }
            }

            if (max_change < tol_) break;
        }

        w_out = w;
        b_out = b;
    }
};

/// @}

} // namespace Skigen

#endif // SKIGEN_LINEAR_MODEL_LOGISTIC_REGRESSION_H
