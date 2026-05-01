// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_METRICS_H
#define SKIGEN_METRICS_H

#include <Eigen/Core>
#include <cmath>
#include <stdexcept>
#include <map>
#include <vector>

namespace Skigen {
namespace Metrics {

// ==================== Regression Metrics ====================

/// Mean Squared Error
template <typename Scalar>
[[nodiscard]] Scalar mean_squared_error(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& y_true,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("y_true and y_pred must have the same size.");
    }
    return (y_true - y_pred).squaredNorm() / static_cast<Scalar>(y_true.size());
}

/// Root Mean Squared Error
template <typename Scalar>
[[nodiscard]] Scalar root_mean_squared_error(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& y_true,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& y_pred) {
    return std::sqrt(mean_squared_error(y_true, y_pred));
}

/// Mean Absolute Error
template <typename Scalar>
[[nodiscard]] Scalar mean_absolute_error(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& y_true,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("y_true and y_pred must have the same size.");
    }
    return (y_true - y_pred).array().abs().sum() / static_cast<Scalar>(y_true.size());
}

/// R² Score (coefficient of determination)
template <typename Scalar>
[[nodiscard]] Scalar r2_score(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& y_true,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("y_true and y_pred must have the same size.");
    }
    Scalar ss_res = (y_true - y_pred).squaredNorm();
    Scalar ss_tot = (y_true.array() - y_true.mean()).matrix().squaredNorm();
    if (ss_tot == Scalar{0}) return Scalar{0};
    return Scalar{1} - ss_res / ss_tot;
}

// ==================== Classification Metrics ====================

/// Accuracy Score
[[nodiscard]] inline double accuracy_score(
    const Eigen::VectorXi& y_true,
    const Eigen::VectorXi& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("y_true and y_pred must have the same size.");
    }
    int correct = 0;
    for (Eigen::Index i = 0; i < y_true.size(); ++i) {
        if (y_true(i) == y_pred(i)) ++correct;
    }
    return static_cast<double>(correct) / static_cast<double>(y_true.size());
}

/// Confusion Matrix — returns (n_classes x n_classes) matrix.
/// Entry (i, j) = number of samples with true label i predicted as j.
[[nodiscard]] inline Eigen::MatrixXi confusion_matrix(
    const Eigen::VectorXi& y_true,
    const Eigen::VectorXi& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("y_true and y_pred must have the same size.");
    }

    // Find unique classes
    std::map<int, int> class_to_idx;
    for (Eigen::Index i = 0; i < y_true.size(); ++i) {
        class_to_idx.emplace(y_true(i), 0);
        class_to_idx.emplace(y_pred(i), 0);
    }
    int idx = 0;
    for (auto& [cls, id] : class_to_idx) id = idx++;

    int n_classes = static_cast<int>(class_to_idx.size());
    Eigen::MatrixXi cm = Eigen::MatrixXi::Zero(n_classes, n_classes);
    for (Eigen::Index i = 0; i < y_true.size(); ++i) {
        cm(class_to_idx[y_true(i)], class_to_idx[y_pred(i)])++;
    }
    return cm;
}

/// Precision (macro-average)
[[nodiscard]] inline double precision_score(
    const Eigen::VectorXi& y_true,
    const Eigen::VectorXi& y_pred) {
    Eigen::MatrixXi cm = confusion_matrix(y_true, y_pred);
    int n_classes = static_cast<int>(cm.rows());
    double total{0};
    int valid_classes = 0;
    for (int k = 0; k < n_classes; ++k) {
        int col_sum = cm.col(k).sum();
        if (col_sum > 0) {
            total += static_cast<double>(cm(k, k)) / static_cast<double>(col_sum);
            ++valid_classes;
        }
    }
    return valid_classes > 0 ? total / static_cast<double>(valid_classes) : 0.0;
}

/// Recall (macro-average)
[[nodiscard]] inline double recall_score(
    const Eigen::VectorXi& y_true,
    const Eigen::VectorXi& y_pred) {
    Eigen::MatrixXi cm = confusion_matrix(y_true, y_pred);
    int n_classes = static_cast<int>(cm.rows());
    double total{0};
    int valid_classes = 0;
    for (int k = 0; k < n_classes; ++k) {
        int row_sum = cm.row(k).sum();
        if (row_sum > 0) {
            total += static_cast<double>(cm(k, k)) / static_cast<double>(row_sum);
            ++valid_classes;
        }
    }
    return valid_classes > 0 ? total / static_cast<double>(valid_classes) : 0.0;
}

/// F1 Score (macro-average)
[[nodiscard]] inline double f1_score(
    const Eigen::VectorXi& y_true,
    const Eigen::VectorXi& y_pred) {
    double p = precision_score(y_true, y_pred);
    double r = recall_score(y_true, y_pred);
    if (p + r == 0.0) return 0.0;
    return 2.0 * p * r / (p + r);
}

// ==================== Pairwise Metrics ====================

/// Euclidean distances: result(i,j) = ||X.row(i) - Y.row(j)||₂
template <typename Scalar>
[[nodiscard]] Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
euclidean_distances(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& X,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& Y) {
    // ||x-y||² = ||x||² + ||y||² - 2*x·y
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    VectorType X_sqnorm = X.rowwise().squaredNorm();
    VectorType Y_sqnorm = Y.rowwise().squaredNorm();
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> D =
        X_sqnorm.replicate(1, Y.rows()) +
        Y_sqnorm.transpose().replicate(X.rows(), 1) -
        Scalar{2} * X * Y.transpose();
    // Clamp negatives from floating-point error
    return D.array().max(Scalar{0}).sqrt().matrix();
}

/// Cosine similarity: result(i,j) = (X.row(i) · Y.row(j)) / (||X.row(i)|| * ||Y.row(j)||)
template <typename Scalar>
[[nodiscard]] Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
cosine_similarity(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& X,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& Y) {
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    VectorType X_norms = X.rowwise().norm();
    VectorType Y_norms = Y.rowwise().norm();

    // Avoid division by zero
    X_norms = X_norms.array().max(std::numeric_limits<Scalar>::epsilon());
    Y_norms = Y_norms.array().max(std::numeric_limits<Scalar>::epsilon());

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> dots = X * Y.transpose();

    for (Eigen::Index i = 0; i < dots.rows(); ++i) {
        for (Eigen::Index j = 0; j < dots.cols(); ++j) {
            dots(i, j) /= X_norms(i) * Y_norms(j);
        }
    }
    return dots;
}

} // namespace Metrics
} // namespace Skigen

#endif // SKIGEN_METRICS_H
