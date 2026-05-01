// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_LINEAR_MODEL_LOGISTIC_REGRESSION_H
#define SKIGEN_LINEAR_MODEL_LOGISTIC_REGRESSION_H

#include "../Core/Validation.h"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <vector>

namespace Skigen {

/// LogisticRegression — Logistic regression classifier (L2, IRLS solver).
/// Supports binary and multiclass (one-vs-rest).
/// Mirrors sklearn.linear_model.LogisticRegression.
template <typename Scalar = double>
class LogisticRegression {
public:
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using RowVectorType = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;
    using IndexVector = Eigen::VectorXi;

    explicit LogisticRegression(Scalar C = Scalar{1}, bool fit_intercept = true,
                                int max_iter = 100, Scalar tol = Scalar{1e-4})
        : C_(C), fit_intercept_(fit_intercept), max_iter_(max_iter), tol_(tol) {}

    [[nodiscard]] bool is_fitted() const noexcept { return fitted_; }

    [[nodiscard]] const MatrixType& coef() const {
        if (!fitted_) throw std::runtime_error("LogisticRegression not fitted.");
        return coef_;
    }
    [[nodiscard]] const VectorType& intercept() const {
        if (!fitted_) throw std::runtime_error("LogisticRegression not fitted.");
        return intercept_;
    }
    [[nodiscard]] const std::vector<int>& classes() const {
        if (!fitted_) throw std::runtime_error("LogisticRegression not fitted.");
        return classes_;
    }

    LogisticRegression& fit(const Eigen::Ref<const MatrixType>& X,
                            const Eigen::Ref<const IndexVector>& y) {
        internal::check_non_empty(X);
        if (X.rows() != y.rows()) {
            throw std::invalid_argument("X and y have inconsistent lengths.");
        }

        n_features_in_ = X.cols();

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

        fitted_ = true;
        return *this;
    }

    [[nodiscard]] IndexVector predict(
        const Eigen::Ref<const MatrixType>& X) const {
        if (!fitted_) throw std::runtime_error("LogisticRegression not fitted.");

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

    [[nodiscard]] MatrixType predict_proba(
        const Eigen::Ref<const MatrixType>& X) const {
        if (!fitted_) throw std::runtime_error("LogisticRegression not fitted.");

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

    [[nodiscard]] Scalar score(const Eigen::Ref<const MatrixType>& X,
                               const Eigen::Ref<const IndexVector>& y) const {
        IndexVector preds = predict(X);
        int correct = 0;
        for (Eigen::Index i = 0; i < y.size(); ++i) {
            if (preds(i) == y(i)) ++correct;
        }
        return static_cast<Scalar>(correct) / static_cast<Scalar>(y.size());
    }

private:
    Scalar C_;
    bool fit_intercept_;
    int max_iter_;
    Scalar tol_;

    bool fitted_ = false;
    Eigen::Index n_features_in_ = 0;
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

} // namespace Skigen

#endif // SKIGEN_LINEAR_MODEL_LOGISTIC_REGRESSION_H
