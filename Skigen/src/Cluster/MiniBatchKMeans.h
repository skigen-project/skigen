// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_CLUSTER_MINI_BATCH_KMEANS_H
#define SKIGEN_CLUSTER_MINI_BATCH_KMEANS_H

#include "../Core/Validation.h"
#include "KMeans.h"

#include <Eigen/Core>
#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace Skigen {

/// @defgroup Algo_MiniBatchKMeans MiniBatchKMeans
/// @ingroup Cluster
/// @brief Mini-batch variant of K-Means for large datasets.
/// @{

/// @brief Mini-Batch K-Means clustering.
///
/// Alternative online implementation of KMeans that uses mini-batches
/// to reduce the computation time, while still attempting to optimise
/// the same objective function.
///
/// Mirrors
/// [sklearn.cluster.MiniBatchKMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html).
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `n_clusters` | `int` | `8` | The number of clusters to form. |
/// | `batch_size` | `int` | `100` | Size of the mini batches. |
/// | `max_iter` | `int` | `100` | Maximum number of iterations over the complete dataset. |
/// | `random_state` | `unsigned int` | `42` | Seed for random sampling. |
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type | Description |
/// |----------|------|-------------|
/// | `cluster_centers()` | `MatrixType` | Coordinates of cluster centers (n_clusters × n_features). |
/// | `labels()` | `IndexVector` | Labels of each training point. |
/// | `inertia()` | `Scalar` | Sum of squared distances to closest center (computed on full data). |
///
/// ### See also
///
/// - Skigen::KMeans — Standard K-Means (more accurate, slower on large data).
///
/// @note **scikit-learn parity gaps:** The following sklearn constructor
///   parameters are not yet supported: `init` (only k-means++), `n_init`,
///   `tol`, `verbose`, `compute_labels`, `max_no_improvement`,
///   `init_size`, `reassignment_ratio`.
///   The following sklearn fitted attributes are not yet exposed:
///   `n_iter_`, `n_steps_`, `n_features_in_`, `feature_names_in_`.
///
/// ### Examples
///
/// @snippet kmeans.cpp example_mini_batch_kmeans
template <typename Scalar = double>
class MiniBatchKMeans {
public:
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using IndexVector = Eigen::VectorXi;

    /// @brief Construct a MiniBatchKMeans estimator.
    ///
    /// @param n_clusters The number of clusters (`int`, default `8`).
    /// @param batch_size Size of the mini batches (`int`, default `100`).
    /// @param max_iter Maximum iterations (`int`, default `100`).
    /// @param random_state RNG seed (`unsigned int`, default `42`).
    explicit MiniBatchKMeans(int n_clusters = 8, int batch_size = 100,
                             int max_iter = 100, unsigned int random_state = 42)
        : n_clusters_(n_clusters), batch_size_(batch_size),
          max_iter_(max_iter), random_state_(random_state) {}

    /// @brief Whether the estimator has been fitted.
    [[nodiscard]] bool is_fitted() const noexcept { return fitted_; }
    /// @brief The number of clusters.
    [[nodiscard]] int n_clusters() const noexcept { return n_clusters_; }

    /// @brief Cluster centers (n_clusters × n_features).
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] const MatrixType& cluster_centers() const {
        if (!fitted_) throw std::runtime_error("MiniBatchKMeans not fitted.");
        return cluster_centers_;
    }
    /// @brief Labels of each training point.
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] const IndexVector& labels() const {
        if (!fitted_) throw std::runtime_error("MiniBatchKMeans not fitted.");
        return labels_;
    }
    /// @brief Sum of squared distances to closest cluster center.
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] Scalar inertia() const {
        if (!fitted_) throw std::runtime_error("MiniBatchKMeans not fitted.");
        return inertia_;
    }

    /// @brief Fit the MiniBatchKMeans model.
    ///
    /// Uses k-means++ initialization on the first batch, then performs
    /// mini-batch stochastic updates to cluster centers.
    ///
    /// @param X Training data of shape (n_samples, n_features).
    /// @return Reference to the fitted estimator (`*this`).
    /// @throws std::invalid_argument if `n_samples < n_clusters`.
    MiniBatchKMeans& fit(const Eigen::Ref<const MatrixType>& X) {
        internal::check_non_empty(X);
        if (X.rows() < n_clusters_) {
            throw std::invalid_argument("n_samples must be >= n_clusters.");
        }

        const Eigen::Index n = X.rows();
        const Eigen::Index actual_batch = std::min(
            static_cast<Eigen::Index>(batch_size_), n);

        std::mt19937 rng(random_state_);

        // Initialize with k-means++ on first batch
        cluster_centers_ = kmeans_plus_plus(X, rng);
        Eigen::VectorXi counts = Eigen::VectorXi::Ones(n_clusters_);

        std::vector<Eigen::Index> all_indices(static_cast<std::size_t>(n));
        std::iota(all_indices.begin(), all_indices.end(), Eigen::Index{0});

        for (int iter = 0; iter < max_iter_; ++iter) {
            // Sample a mini-batch
            std::shuffle(all_indices.begin(), all_indices.end(), rng);

            MatrixType batch(actual_batch, X.cols());
            for (Eigen::Index i = 0; i < actual_batch; ++i) {
                batch.row(i) = X.row(all_indices[static_cast<std::size_t>(i)]);
            }

            // Assign batch points to nearest centers
            IndexVector batch_labels(actual_batch);
            for (Eigen::Index i = 0; i < actual_batch; ++i) {
                Scalar best_dist = std::numeric_limits<Scalar>::max();
                int best_k = 0;
                for (int k = 0; k < n_clusters_; ++k) {
                    Scalar d = (batch.row(i) - cluster_centers_.row(k)).squaredNorm();
                    if (d < best_dist) { best_dist = d; best_k = k; }
                }
                batch_labels(i) = best_k;
            }

            // Update centers with streaming mean
            for (Eigen::Index i = 0; i < actual_batch; ++i) {
                int k = batch_labels(i);
                counts(k) += 1;
                Scalar lr = Scalar{1} / static_cast<Scalar>(counts(k));
                cluster_centers_.row(k) +=
                    lr * (batch.row(i) - cluster_centers_.row(k));
            }
        }

        // Final assignment for all points
        labels_.resize(n);
        inertia_ = Scalar{0};
        for (Eigen::Index i = 0; i < n; ++i) {
            Scalar best_dist = std::numeric_limits<Scalar>::max();
            int best_k = 0;
            for (int k = 0; k < n_clusters_; ++k) {
                Scalar d = (X.row(i) - cluster_centers_.row(k)).squaredNorm();
                if (d < best_dist) { best_dist = d; best_k = k; }
            }
            labels_(i) = best_k;
            inertia_ += best_dist;
        }

        fitted_ = true;
        return *this;
    }

    /// @brief Predict the closest cluster each sample belongs to.
    ///
    /// @param X New data of shape (n_samples, n_features).
    /// @return Index of the closest cluster for each sample.
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] IndexVector predict(const Eigen::Ref<const MatrixType>& X) const {
        if (!fitted_) throw std::runtime_error("MiniBatchKMeans not fitted.");
        IndexVector labels(X.rows());
        for (Eigen::Index i = 0; i < X.rows(); ++i) {
            Scalar best_dist = std::numeric_limits<Scalar>::max();
            int best_k = 0;
            for (int k = 0; k < n_clusters_; ++k) {
                Scalar d = (X.row(i) - cluster_centers_.row(k)).squaredNorm();
                if (d < best_dist) { best_dist = d; best_k = k; }
            }
            labels(i) = best_k;
        }
        return labels;
    }

private:
    int n_clusters_;
    int batch_size_;
    int max_iter_;
    unsigned int random_state_;

    bool fitted_ = false;
    MatrixType cluster_centers_;
    IndexVector labels_;
    Scalar inertia_ = Scalar{0};

    MatrixType kmeans_plus_plus(const Eigen::Ref<const MatrixType>& X,
                                std::mt19937& rng) const {
        const Eigen::Index n = X.rows();
        MatrixType centers(n_clusters_, X.cols());

        std::uniform_int_distribution<Eigen::Index> first_dist(0, n - 1);
        centers.row(0) = X.row(first_dist(rng));

        VectorType min_dist = VectorType::Constant(n, std::numeric_limits<Scalar>::max());

        for (int k = 1; k < n_clusters_; ++k) {
            for (Eigen::Index i = 0; i < n; ++i) {
                Scalar d = (X.row(i) - centers.row(k - 1)).squaredNorm();
                min_dist(i) = std::min(min_dist(i), d);
            }
            std::discrete_distribution<Eigen::Index> dist(
                min_dist.data(), min_dist.data() + n);
            centers.row(k) = X.row(dist(rng));
        }

        return centers;
    }
};

/// @}

} // namespace Skigen

#endif // SKIGEN_CLUSTER_MINI_BATCH_KMEANS_H
