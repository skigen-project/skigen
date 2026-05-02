// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_NEIGHBORS_KNEIGHBORS_CLASSIFIER_H
#define SKIGEN_NEIGHBORS_KNEIGHBORS_CLASSIFIER_H

#include "../Core/Base.h"
#include "../Core/Validation.h"

#include <Eigen/Core>
#include <algorithm>
#include <map>
#include <vector>

namespace Skigen {

/// @defgroup Algo_KNeighbors K-Nearest Neighbors
/// @ingroup Neighbors
/// @brief Brute-force k-nearest neighbors classifiers and regressors.
/// @{

// ---------------------------------------------------------------------------
// Internal: shared brute-force k-NN distance computation
// ---------------------------------------------------------------------------

namespace internal {

/// @brief Compute indices of the k nearest training rows to each query row.
///
/// Returns a matrix of shape (n_queries, k) with training-set indices
/// sorted by ascending squared Euclidean distance.
template <typename Scalar>
Eigen::MatrixXi kneighbors_indices(
    const Eigen::Ref<const Eigen::Matrix<Scalar, Eigen::Dynamic,
                                         Eigen::Dynamic>>& X_train,
    const Eigen::Ref<const Eigen::Matrix<Scalar, Eigen::Dynamic,
                                         Eigen::Dynamic>>& X_query,
    int k) {
    const auto n_train = X_train.rows();
    const auto n_query = X_query.rows();
    Eigen::MatrixXi result(n_query, k);

    std::vector<std::pair<Scalar, int>> dists(
        static_cast<std::size_t>(n_train));

    for (Eigen::Index i = 0; i < n_query; ++i) {
        for (Eigen::Index j = 0; j < n_train; ++j) {
            dists[static_cast<std::size_t>(j)] = {
                (X_query.row(i) - X_train.row(j)).squaredNorm(),
                static_cast<int>(j)};
        }
        std::partial_sort(dists.begin(), dists.begin() + k, dists.end());
        for (int ki = 0; ki < k; ++ki) {
            result(i, ki) = dists[static_cast<std::size_t>(ki)].second;
        }
    }
    return result;
}

} // namespace internal

// ---------------------------------------------------------------------------
// KNeighborsClassifier
// ---------------------------------------------------------------------------

/// @brief Classifier implementing the k-nearest neighbors vote.
///
/// Uses brute-force computation of squared Euclidean distances. For each
/// query point, finds the `n_neighbors` closest training points and
/// predicts by majority vote.
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
/// ### Attributes (after fitting)
///
/// | Accessor | Type | Description |
/// |----------|------|-------------|
/// | `n_neighbors()` | `int` | Number of neighbors. |
/// | `is_fitted()` | `bool` | Whether the estimator has been fitted. |
/// | `n_features_in()` | `IndexType` | Number of features seen during fit. |
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
///
/// ### Examples
///
/// @snippet kneighbors.cpp example_kneighbors_classifier
template <typename Scalar = double>
class KNeighborsClassifier
    : public Classifier<KNeighborsClassifier<Scalar>, Scalar> {
public:
    using Base = Classifier<KNeighborsClassifier<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::IndexType;
    using typename Base::ScalarType;
    using typename Base::LabelType;

    /// @brief Construct a KNeighborsClassifier.
    ///
    /// @param n_neighbors Number of neighbors to use (`int`, default `5`).
    explicit KNeighborsClassifier(int n_neighbors = 5)
        : n_neighbors_(n_neighbors) {}

    // -- Accessors ----------------------------------------------------------

    /// @brief Number of neighbors.
    [[nodiscard]] int n_neighbors() const noexcept { return n_neighbors_; }

    // -- Implementation (called by CRTP base) --------------------------------

    /// @brief Fit the k-nearest neighbors classifier from the training set.
    KNeighborsClassifier& fit_impl(
        const Eigen::Ref<const MatrixType>& X,
        const Eigen::Ref<const LabelType>& y) {
        internal::check_non_empty(X);
        internal::check_consistent_length(X, y);
        if (X.rows() < n_neighbors_) {
            throw std::invalid_argument(
                "n_samples (" + std::to_string(X.rows()) +
                ") must be >= n_neighbors (" +
                std::to_string(n_neighbors_) + ").");
        }

        X_train_ = X;
        y_train_ = y;
        this->n_features_in_ = X.cols();
        this->fitted_ = true;
        return *this;
    }

    /// @brief Predict class labels for the provided data.
    [[nodiscard]] LabelType predict_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        auto idx = internal::kneighbors_indices<Scalar>(
            X_train_, X, n_neighbors_);

        LabelType predictions(X.rows());
        for (Eigen::Index i = 0; i < X.rows(); ++i) {
            std::map<int, int> votes;
            for (int ki = 0; ki < n_neighbors_; ++ki) {
                votes[y_train_(idx(i, ki))]++;
            }
            int best_label = 0, best_count = 0;
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

private:
    int n_neighbors_;
    MatrixType X_train_;
    LabelType y_train_;
};

// ---------------------------------------------------------------------------
// KNeighborsRegressor
// ---------------------------------------------------------------------------

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
/// ### Attributes (after fitting)
///
/// | Accessor | Type | Description |
/// |----------|------|-------------|
/// | `n_neighbors()` | `int` | Number of neighbors. |
/// | `is_fitted()` | `bool` | Whether the estimator has been fitted. |
/// | `n_features_in()` | `IndexType` | Number of features seen during fit. |
///
/// @note **scikit-learn parity gaps:** Same as KNeighborsClassifier —
///   `weights`, `algorithm`, `leaf_size`, `p`, `metric`, etc.
///   are not yet supported.
///
/// ### Examples
///
/// @snippet kneighbors.cpp example_kneighbors_regressor
template <typename Scalar = double>
class KNeighborsRegressor
    : public Predictor<KNeighborsRegressor<Scalar>, Scalar> {
public:
    using Base = Predictor<KNeighborsRegressor<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::IndexType;
    using typename Base::ScalarType;

    /// @brief Construct a KNeighborsRegressor.
    ///
    /// @param n_neighbors Number of neighbors to use (`int`, default `5`).
    explicit KNeighborsRegressor(int n_neighbors = 5)
        : n_neighbors_(n_neighbors) {}

    // -- Accessors ----------------------------------------------------------

    /// @brief Number of neighbors.
    [[nodiscard]] int n_neighbors() const noexcept { return n_neighbors_; }

    // -- Implementation (called by CRTP base) --------------------------------

    /// @brief Fit the k-nearest neighbors regressor from the training set.
    KNeighborsRegressor& fit_impl(
        const Eigen::Ref<const MatrixType>& X,
        const Eigen::Ref<const VectorType>& y) {
        internal::check_non_empty(X);
        internal::check_consistent_length(X, y);
        if (X.rows() < n_neighbors_) {
            throw std::invalid_argument(
                "n_samples (" + std::to_string(X.rows()) +
                ") must be >= n_neighbors (" +
                std::to_string(n_neighbors_) + ").");
        }

        X_train_ = X;
        y_train_ = y;
        this->n_features_in_ = X.cols();
        this->fitted_ = true;
        return *this;
    }

    /// @brief Predict target values for the provided data.
    [[nodiscard]] VectorType predict_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        auto idx = internal::kneighbors_indices<Scalar>(
            X_train_, X, n_neighbors_);

        VectorType predictions(X.rows());
        for (Eigen::Index i = 0; i < X.rows(); ++i) {
            Scalar sum{0};
            for (int ki = 0; ki < n_neighbors_; ++ki) {
                sum += y_train_(idx(i, ki));
            }
            predictions(i) = sum / static_cast<Scalar>(n_neighbors_);
        }
        return predictions;
    }

    /// @brief Return the @f$R^2@f$ coefficient of determination.
    [[nodiscard]] ScalarType score_impl(
        const Eigen::Ref<const MatrixType>& X,
        const Eigen::Ref<const VectorType>& y) const {
        VectorType y_pred = predict_impl(X);
        Scalar ss_res = (y - y_pred).squaredNorm();
        Scalar ss_tot = (y.array() - y.mean()).matrix().squaredNorm();
        if (ss_tot == Scalar{0}) return Scalar{0};
        return Scalar{1} - ss_res / ss_tot;
    }

private:
    int n_neighbors_;
    MatrixType X_train_;
    VectorType y_train_;
};

/// @}

} // namespace Skigen

#endif // SKIGEN_NEIGHBORS_KNEIGHBORS_CLASSIFIER_H
