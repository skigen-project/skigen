// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_NEIGHBORS_KNEIGHBORS_CLASSIFIER_H
#define SKIGEN_NEIGHBORS_KNEIGHBORS_CLASSIFIER_H

#include "../Core/Validation.h"

#include <Eigen/Core>
#include <algorithm>
#include <map>
#include <stdexcept>
#include <vector>

namespace Skigen {

/// @defgroup Algo_KNeighbors K-Nearest Neighbors
/// @ingroup Neighbors
/// @brief Brute-force k-nearest neighbors classifiers and regressors.
/// @{

/// @brief Classifier implementing the k-nearest neighbors vote.
///
/// Uses brute-force computation of distances. For each query point,
/// finds the `n_neighbors` closest training points and predicts by
/// majority vote.
///
/// Mirrors
/// [sklearn.neighbors.KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html).
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `n_neighbors` | `int` | `5` | Number of neighbors to use. |
///
/// ### Notes
///
/// Only brute-force (Euclidean distance) is implemented. This is
/// @f$O(n \cdot m)@f$ per query where @f$n@f$ is the number of
/// training samples and @f$m@f$ is the number of features.
///
/// @note **scikit-learn parity gaps:** The following sklearn constructor
///   parameters are not yet supported: `weights`, `algorithm`
///   (only brute-force), `leaf_size`, `p`, `metric`, `metric_params`,
///   `n_jobs`.
///   The following sklearn fitted attributes are not yet exposed:
///   `classes_`, `effective_metric_`, `effective_metric_params_`,
///   `n_features_in_`, `feature_names_in_`, `n_samples_fit_`,
///   `outputs_2d_`.
///
/// ### Examples
///
/// @snippet kneighbors.cpp example_kneighbors_classifier
template <typename Scalar = double>
class KNeighborsClassifier {
public:
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using IndexVector = Eigen::VectorXi;

    /// @brief Construct a KNeighborsClassifier.
    ///
    /// @param n_neighbors Number of neighbors to use (`int`, default `5`).
    explicit KNeighborsClassifier(int n_neighbors = 5)
        : n_neighbors_(n_neighbors) {}

    // -- Accessors ----------------------------------------------------------

    /// @brief Whether the estimator has been fitted.
    [[nodiscard]] bool is_fitted() const noexcept { return fitted_; }
    /// @brief Number of neighbors.
    [[nodiscard]] int n_neighbors() const noexcept { return n_neighbors_; }

    // -- fit / predict -------------------------------------------------------

    /// @brief Fit the k-nearest neighbors classifier from the training set.
    ///
    /// Stores the training data for later queries.
    ///
    /// @param X Training data of shape (n_samples, n_features).
    /// @param y Target values of shape (n_samples,) with integer class labels.
    /// @return Reference to the fitted estimator (`*this`).
    /// @throws std::invalid_argument if `n_samples < n_neighbors` or
    ///   X and y have inconsistent lengths.
    KNeighborsClassifier& fit(const Eigen::Ref<const MatrixType>& X,
                              const Eigen::Ref<const IndexVector>& y) {
        internal::check_non_empty(X);
        if (X.rows() != y.rows()) {
            throw std::invalid_argument("X and y have inconsistent lengths.");
        }
        if (X.rows() < n_neighbors_) {
            throw std::invalid_argument(
                "n_samples must be >= n_neighbors.");
        }

        X_train_ = X;
        y_train_ = y;
        n_features_in_ = X.cols();
        fitted_ = true;
        return *this;
    }

    /// @brief Predict class labels for the provided data.
    ///
    /// @param X Test samples of shape (n_samples, n_features).
    /// @return Integer vector of predicted class labels (n_samples,).
    /// @throws std::runtime_error if the model has not been fitted.
    /// @throws std::invalid_argument if feature count doesn't match training data.
    [[nodiscard]] IndexVector predict(
        const Eigen::Ref<const MatrixType>& X) const {
        if (!fitted_) throw std::runtime_error(
            "KNeighborsClassifier has not been fitted yet.");
        if (X.cols() != n_features_in_) {
            throw std::invalid_argument("Feature count mismatch.");
        }

        IndexVector predictions(X.rows());

        // For each query point, find k nearest neighbors and vote
        std::vector<std::pair<Scalar, int>> dists(
            static_cast<std::size_t>(X_train_.rows()));

        for (Eigen::Index i = 0; i < X.rows(); ++i) {
            for (Eigen::Index j = 0; j < X_train_.rows(); ++j) {
                dists[static_cast<std::size_t>(j)] = {
                    (X.row(i) - X_train_.row(j)).squaredNorm(),
                    y_train_(j)
                };
            }

            std::partial_sort(dists.begin(),
                              dists.begin() + n_neighbors_,
                              dists.end());

            // Majority vote
            std::map<int, int> votes;
            for (int k = 0; k < n_neighbors_; ++k) {
                votes[dists[static_cast<std::size_t>(k)].second]++;
            }

            int best_label = 0;
            int best_count = 0;
            for (const auto& [label, count] : votes) {
                if (count > best_count) {
                    best_count = count;
                    best_label = label;
                }
            }
            predictions(i) = best_label;
        }

        return predictions;
    }

    /// @brief Return the mean accuracy on the given test data and labels.
    ///
    /// @param X Test samples of shape (n_samples, n_features).
    /// @param y True class labels of shape (n_samples,).
    /// @return Mean accuracy.
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
    int n_neighbors_;
    bool fitted_ = false;
    Eigen::Index n_features_in_ = 0;

    MatrixType X_train_;
    IndexVector y_train_;
};

/// @brief Regression based on k-nearest neighbors.
///
/// The target is predicted by local interpolation of the targets
/// associated of the nearest neighbors in the training set (mean
/// of the k nearest values).
///
/// Mirrors
/// [sklearn.neighbors.KNeighborsRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html).
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `n_neighbors` | `int` | `5` | Number of neighbors to use. |
///
/// @note **scikit-learn parity gaps:** Same as KNeighborsClassifier —
///   `weights`, `algorithm`, `leaf_size`, `p`, `metric`, etc.
///   are not yet supported.
///
/// ### Examples
///
/// @snippet kneighbors.cpp example_kneighbors_regressor
template <typename Scalar = double>
class KNeighborsRegressor {
public:
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    /// @brief Construct a KNeighborsRegressor.
    ///
    /// @param n_neighbors Number of neighbors to use (`int`, default `5`).
    explicit KNeighborsRegressor(int n_neighbors = 5)
        : n_neighbors_(n_neighbors) {}

    /// @brief Whether the estimator has been fitted.
    [[nodiscard]] bool is_fitted() const noexcept { return fitted_; }

    /// @brief Fit the k-nearest neighbors regressor.
    ///
    /// @param X Training data of shape (n_samples, n_features).
    /// @param y Target values of shape (n_samples,).
    /// @return Reference to the fitted estimator (`*this`).
    /// @throws std::invalid_argument if X and y have inconsistent lengths.
    KNeighborsRegressor& fit(const Eigen::Ref<const MatrixType>& X,
                             const Eigen::Ref<const VectorType>& y) {
        internal::check_non_empty(X);
        if (X.rows() != y.rows()) {
            throw std::invalid_argument("X and y have inconsistent lengths.");
        }

        X_train_ = X;
        y_train_ = y;
        n_features_in_ = X.cols();
        fitted_ = true;
        return *this;
    }

    /// @brief Predict target values for the provided data.
    ///
    /// Returns the mean of the target values of the k nearest neighbors.
    ///
    /// @param X Test samples of shape (n_samples, n_features).
    /// @return Predicted values of shape (n_samples,).
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] VectorType predict(
        const Eigen::Ref<const MatrixType>& X) const {
        if (!fitted_) throw std::runtime_error(
            "KNeighborsRegressor has not been fitted yet.");

        VectorType predictions(X.rows());
        std::vector<std::pair<Scalar, Scalar>> dists(
            static_cast<std::size_t>(X_train_.rows()));

        for (Eigen::Index i = 0; i < X.rows(); ++i) {
            for (Eigen::Index j = 0; j < X_train_.rows(); ++j) {
                dists[static_cast<std::size_t>(j)] = {
                    (X.row(i) - X_train_.row(j)).squaredNorm(),
                    y_train_(j)
                };
            }

            std::partial_sort(dists.begin(),
                              dists.begin() + n_neighbors_,
                              dists.end());

            Scalar sum{0};
            for (int k = 0; k < n_neighbors_; ++k) {
                sum += dists[static_cast<std::size_t>(k)].second;
            }
            predictions(i) = sum / static_cast<Scalar>(n_neighbors_);
        }

        return predictions;
    }

    /// @brief Return the @f$R^2@f$ coefficient of determination.
    ///
    /// @param X Test samples of shape (n_samples, n_features).
    /// @param y True values of shape (n_samples,).
    /// @return @f$R^2@f$ score.
    [[nodiscard]] Scalar score(const Eigen::Ref<const MatrixType>& X,
                               const Eigen::Ref<const VectorType>& y) const {
        VectorType y_pred = predict(X);
        Scalar ss_res = (y - y_pred).squaredNorm();
        Scalar ss_tot = (y.array() - y.mean()).matrix().squaredNorm();
        if (ss_tot == Scalar{0}) return Scalar{0};
        return Scalar{1} - ss_res / ss_tot;
    }

private:
    int n_neighbors_;
    bool fitted_ = false;
    Eigen::Index n_features_in_ = 0;

    MatrixType X_train_;
    VectorType y_train_;
};

/// @}

} // namespace Skigen

#endif // SKIGEN_NEIGHBORS_KNEIGHBORS_CLASSIFIER_H
