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

/// MiniBatchKMeans — Mini-batch variant of K-Means.
/// Mirrors sklearn.cluster.MiniBatchKMeans.
template <typename Scalar = double>
class MiniBatchKMeans {
public:
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using IndexVector = Eigen::VectorXi;

    explicit MiniBatchKMeans(int n_clusters = 8, int batch_size = 100,
                             int max_iter = 100, unsigned int random_state = 42)
        : n_clusters_(n_clusters), batch_size_(batch_size),
          max_iter_(max_iter), random_state_(random_state) {}

    [[nodiscard]] bool is_fitted() const noexcept { return fitted_; }
    [[nodiscard]] int n_clusters() const noexcept { return n_clusters_; }

    [[nodiscard]] const MatrixType& cluster_centers() const {
        if (!fitted_) throw std::runtime_error("MiniBatchKMeans not fitted.");
        return cluster_centers_;
    }
    [[nodiscard]] const IndexVector& labels() const {
        if (!fitted_) throw std::runtime_error("MiniBatchKMeans not fitted.");
        return labels_;
    }
    [[nodiscard]] Scalar inertia() const {
        if (!fitted_) throw std::runtime_error("MiniBatchKMeans not fitted.");
        return inertia_;
    }

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

} // namespace Skigen

#endif // SKIGEN_CLUSTER_MINI_BATCH_KMEANS_H
