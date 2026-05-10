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
/// ### Limitations relative to scikit-learn
///
/// The following scikit-learn constructor
///   parameters are not honoured: `weights`, `algorithm`
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
/// ### Limitations relative to scikit-learn Same as KNeighborsClassifier —
///   `weights`, `algorithm`, `leaf_size`, `p`, `metric`, etc.
///   are not honoured.
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
        // Invalidate any prior multi-target state.
        Y_train_.resize(0, 0);
        y_train_matrix_view_.resize(0, 0);
        n_targets_ = 1;
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

    // -- Multi-target regression ------------------------------

    /// @brief Fit on multi-target Y (shape n × n_targets).
    ///
    /// Stores X and Y; prediction averages neighbour rows of `Y_train_`
    /// per target. Single-target API is unchanged — `predict()` keeps
    /// returning a 1-D vector reflecting the first target column when
    /// `fit_multi` was used.
    KNeighborsRegressor& fit_multi(
        const Eigen::Ref<const MatrixType>& X,
        const Eigen::Ref<const MatrixType>& Y) {
        internal::check_non_empty(X);
        if (X.rows() != Y.rows()) {
            throw std::invalid_argument(
                "fit_multi: X has " + std::to_string(X.rows()) +
                " rows but Y has " + std::to_string(Y.rows()) + ".");
        }
        if (Y.cols() < 1) {
            throw std::invalid_argument(
                "fit_multi: Y must have at least 1 target column.");
        }
        if (X.rows() < n_neighbors_) {
            throw std::invalid_argument(
                "n_samples (" + std::to_string(X.rows()) +
                ") must be >= n_neighbors (" +
                std::to_string(n_neighbors_) + ").");
        }
        X_train_ = X;
        Y_train_ = Y;
        y_train_ = Y.col(0);                  // mirror first target
        n_targets_ = static_cast<int>(Y.cols());
        this->n_features_in_ = X.cols();
        this->fitted_ = true;
        return *this;
    }

    /// @brief Predict (n_samples × n_targets) — mean of neighbour rows
    ///   from `Y_train_` per target.
    [[nodiscard]] MatrixType predict_multi(
        const Eigen::Ref<const MatrixType>& X) const {
        this->check_is_fitted();
        this->validate_feature_count(X);
        auto idx = internal::kneighbors_indices<Scalar>(
            X_train_, X, n_neighbors_);
        // If Y_train_ is empty (single-target fit was used), synthesise a
        // 1-column matrix from y_train_.
        const MatrixType& Yref =
            (Y_train_.size() == 0) ? lazy_y_matrix() : Y_train_;
        const Eigen::Index t = Yref.cols();
        MatrixType out = MatrixType::Zero(X.rows(), t);
        for (Eigen::Index i = 0; i < X.rows(); ++i) {
            for (int ki = 0; ki < n_neighbors_; ++ki) {
                out.row(i) += Yref.row(idx(i, ki));
            }
            out.row(i) /= static_cast<Scalar>(n_neighbors_);
        }
        return out;
    }

    [[nodiscard]] int n_targets() const {
        this->check_is_fitted();
        return Y_train_.size() == 0 ? 1 : n_targets_;
    }

private:
    int n_neighbors_;
    MatrixType X_train_;
    VectorType y_train_;

    // Multi-target storage. When `fit_multi` was used, Y_train_ is
    // (n_samples × n_targets); otherwise empty.
    MatrixType Y_train_;
    int        n_targets_ = 1;
    mutable MatrixType y_train_matrix_view_;

    const MatrixType& lazy_y_matrix() const {
        if (y_train_matrix_view_.size() == 0) {
            y_train_matrix_view_.resize(y_train_.size(), 1);
            y_train_matrix_view_.col(0) = y_train_;
        }
        return y_train_matrix_view_;
    }
};

/// @}

} // namespace Skigen

#endif // SKIGEN_NEIGHBORS_KNEIGHBORS_CLASSIFIER_H
