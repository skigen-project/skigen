// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_CLUSTER_MINI_BATCH_KMEANS_H
#define SKIGEN_CLUSTER_MINI_BATCH_KMEANS_H

#include "../Core/Base.h"
#include "../Core/Validation.h"
#include "KMeans.h"

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
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
/// ### Limitations relative to scikit-learn
///
/// The following scikit-learn constructor
///   parameters are not honoured: `init` (only k-means++), `n_init`,
///   `tol`, `verbose`, `compute_labels`, `max_no_improvement`,
///   `init_size`, `reassignment_ratio`.
///   The following sklearn fitted attributes are not exposed:
///   `n_iter_`, `n_steps_`, `n_features_in_`, `feature_names_in_`.
///
/// ### Examples
///
/// @snippet kmeans.cpp example_mini_batch_kmeans
template <typename Scalar = double>
class MiniBatchKMeans : public Estimator<MiniBatchKMeans<Scalar>, Scalar> {
public:
    using Base = Estimator<MiniBatchKMeans<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::IndexType;
    using typename Base::ScalarType;
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
    /// @brief The number of clusters.
    [[nodiscard]] int n_clusters() const noexcept { return n_clusters_; }

    /// @brief Cluster centers (n_clusters × n_features).
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] const MatrixType& cluster_centers() const {
        this->check_is_fitted();
        return cluster_centers_;
    }
    /// @brief Labels of each training point.
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] const IndexVector& labels() const {
        this->check_is_fitted();
        return labels_;
    }
    /// @brief Sum of squared distances to closest cluster center.
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] Scalar inertia() const {
        this->check_is_fitted();
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
            throw std::invalid_argument(
                "n_samples (" + std::to_string(X.rows()) +
                ") must be >= n_clusters (" +
                std::to_string(n_clusters_) + ").");
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

        this->n_features_in_ = X.cols();
        this->fitted_ = true;
        return *this;
    }

    /// @brief Online update of the cluster centers from a single batch.
    ///
    /// Mirrors sklearn's `MiniBatchKMeans.partial_fit`. The first call
    /// initialises centers via k-means++ on the supplied batch (which must
    /// therefore contain at least `n_clusters` samples); subsequent calls
    /// perform a single streaming pass over `X`, updating each assigned
    /// center's running mean.
    ///
    /// Unlike `fit`, `partial_fit` does **not** populate `labels_` or
    /// `inertia_` (matching sklearn behaviour — those attributes refer to
    /// the last `fit` call only).
    ///
    /// @param X Batch of training data, shape (n_samples_batch, n_features).
    /// @return Reference to the fitted estimator (`*this`).
    /// @throws std::invalid_argument on first call if `n_samples_batch <
    ///   n_clusters`, or on subsequent calls if the feature count differs.
    MiniBatchKMeans& partial_fit(const Eigen::Ref<const MatrixType>& X) {
        internal::check_non_empty(X);

        if (!this->fitted_) {
            if (X.rows() < n_clusters_) {
                throw std::invalid_argument(
                    "partial_fit: first batch must contain at least "
                    "n_clusters (" + std::to_string(n_clusters_) +
                    ") samples; got " + std::to_string(X.rows()) + ".");
            }
            this->n_features_in_ = X.cols();
            std::mt19937 rng(random_state_);
            cluster_centers_ = kmeans_plus_plus(X, rng);
            cluster_counts_  = Eigen::VectorXi::Ones(n_clusters_);
            this->fitted_ = true;
        } else {
            if (X.cols() != this->n_features_in_) {
                throw std::invalid_argument(
                    "X has " + std::to_string(X.cols()) +
                    " features, but partial_fit was previously called with " +
                    std::to_string(this->n_features_in_) + " features.");
            }
        }

        for (Eigen::Index i = 0; i < X.rows(); ++i) {
            Scalar best = std::numeric_limits<Scalar>::max();
            int best_k = 0;
            for (int k = 0; k < n_clusters_; ++k) {
                const Scalar d =
                    (X.row(i) - cluster_centers_.row(k)).squaredNorm();
                if (d < best) { best = d; best_k = k; }
            }
            cluster_counts_(best_k) += 1;
            const Scalar lr =
                Scalar{1} / static_cast<Scalar>(cluster_counts_(best_k));
            cluster_centers_.row(best_k) +=
                lr * (X.row(i) - cluster_centers_.row(best_k));
        }
        return *this;
    }

    /// @brief Online update from a sparse mini-batch (see dense overload).
    ///
    /// Mirrors sklearn's `MiniBatchKMeans.partial_fit` on sparse input. Each
    /// row's nearest centroid is found via the
    /// @f$ \|x\|^2 + \|c\|^2 - 2\,x\!\cdot\!c @f$ expansion (sparse dot,
    /// dense centroid); the assigned centroid is then updated by a running
    /// mean using `cluster_counts_`.
    template <int Options, typename StorageIndex>
    MiniBatchKMeans& partial_fit(
        const Eigen::SparseMatrix<Scalar, Options, StorageIndex>& X) {
        if (X.rows() == 0 || X.cols() == 0) {
            throw std::invalid_argument(
                "partial_fit: empty sparse matrix.");
        }

        using RowSparse =
            Eigen::SparseMatrix<Scalar, Eigen::RowMajor, StorageIndex>;
        const RowSparse Xr = X;
        const Eigen::Index n = Xr.rows();

        if (!this->fitted_) {
            if (n < n_clusters_) {
                throw std::invalid_argument(
                    "partial_fit: first batch must contain at least "
                    "n_clusters (" + std::to_string(n_clusters_) +
                    ") samples; got " + std::to_string(n) + ".");
            }
            this->n_features_in_ = X.cols();
            std::mt19937 rng(random_state_);
            // Use the first row materialised dense as the seed centroid;
            // grow with farthest-first via the sparse distance expansion.
            VectorType row_sq_norm(n);
            for (Eigen::Index i = 0; i < n; ++i) {
                Scalar s{0};
                for (typename RowSparse::InnerIterator it(Xr, i); it; ++it) {
                    s += it.value() * it.value();
                }
                row_sq_norm(i) = s;
            }
            cluster_centers_ = MatrixType::Zero(n_clusters_, X.cols());
            std::uniform_int_distribution<Eigen::Index> first_dist(0, n - 1);
            const Eigen::Index first = first_dist(rng);
            for (typename RowSparse::InnerIterator it(Xr, first); it; ++it) {
                cluster_centers_(0, it.col()) = it.value();
            }
            VectorType min_dist =
                VectorType::Constant(n, std::numeric_limits<Scalar>::max());
            for (int k = 1; k < n_clusters_; ++k) {
                const Scalar c_sq =
                    cluster_centers_.row(k - 1).squaredNorm();
                for (Eigen::Index i = 0; i < n; ++i) {
                    Scalar dot{0};
                    for (typename RowSparse::InnerIterator it(Xr, i); it; ++it) {
                        dot += it.value() * cluster_centers_(k - 1, it.col());
                    }
                    const Scalar d =
                        row_sq_norm(i) + c_sq - Scalar{2} * dot;
                    if (d < min_dist(i)) min_dist(i) = d;
                }
                std::discrete_distribution<Eigen::Index> dist(
                    min_dist.data(), min_dist.data() + n);
                const Eigen::Index pick = dist(rng);
                cluster_centers_.row(k).setZero();
                for (typename RowSparse::InnerIterator it(Xr, pick); it; ++it) {
                    cluster_centers_(k, it.col()) = it.value();
                }
            }
            cluster_counts_ = Eigen::VectorXi::Ones(n_clusters_);
            this->fitted_ = true;
        } else {
            if (X.cols() != this->n_features_in_) {
                throw std::invalid_argument(
                    "X has " + std::to_string(X.cols()) +
                    " features, but partial_fit was previously called with " +
                    std::to_string(this->n_features_in_) + " features.");
            }
        }

        for (Eigen::Index i = 0; i < n; ++i) {
            Scalar row_sq{0};
            for (typename RowSparse::InnerIterator it(Xr, i); it; ++it) {
                row_sq += it.value() * it.value();
            }
            Scalar best = std::numeric_limits<Scalar>::max();
            int best_k = 0;
            for (int k = 0; k < n_clusters_; ++k) {
                Scalar dot{0};
                for (typename RowSparse::InnerIterator it(Xr, i); it; ++it) {
                    dot += it.value() * cluster_centers_(k, it.col());
                }
                const Scalar c_sq = cluster_centers_.row(k).squaredNorm();
                const Scalar d = row_sq + c_sq - Scalar{2} * dot;
                if (d < best) { best = d; best_k = k; }
            }
            cluster_counts_(best_k) += 1;
            const Scalar lr =
                Scalar{1} / static_cast<Scalar>(cluster_counts_(best_k));
            // Sparse update: row_dense - center; we apply
            //   center += lr * (row - center) = (1 - lr)*center + lr*row
            // For the dense-row analogue we'd compute it elementwise, but
            // for sparse we apply the (1 - lr) scaling and then add lr*row
            // along the nonzeros only.
            cluster_centers_.row(best_k) *= (Scalar{1} - lr);
            for (typename RowSparse::InnerIterator it(Xr, i); it; ++it) {
                cluster_centers_(best_k, it.col()) += lr * it.value();
            }
        }
        return *this;
    }

    /// @brief Predict the closest cluster each sample belongs to.
    ///
    /// @param X New data of shape (n_samples, n_features).
    /// @return Index of the closest cluster for each sample.
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] IndexVector predict(const Eigen::Ref<const MatrixType>& X) const {
        this->check_is_fitted();
        this->validate_feature_count(X);
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

    MatrixType cluster_centers_;
    Eigen::VectorXi cluster_counts_;   ///< Per-center sample count for partial_fit running mean.
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
