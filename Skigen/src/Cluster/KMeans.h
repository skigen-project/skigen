// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_CLUSTER_KMEANS_H
#define SKIGEN_CLUSTER_KMEANS_H

#include "../Core/Validation.h"

#include <Eigen/Core>
#include <algorithm>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace Skigen {

/// @defgroup Algo_KMeans KMeans
/// @ingroup Cluster
/// @brief K-Means clustering via Lloyd's algorithm with k-means++ initialization.
/// @{

/// @brief K-Means clustering.
///
/// The KMeans algorithm clusters data by trying to separate samples
/// in `n_clusters` groups of equal variance, minimizing the within-cluster
/// sum-of-squares (inertia). Uses k-means++ initialization and
/// Lloyd's iterative algorithm.
///
/// Mirrors
/// [sklearn.cluster.KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html).
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `n_clusters` | `int` | `8` | The number of clusters to form. |
/// | `max_iter` | `int` | `300` | Maximum number of iterations of Lloyd's algorithm for a single run. |
/// | `n_init` | `int` | `10` | Number of times the algorithm is run with different centroid seeds. |
/// | `random_state` | `unsigned int` | `42` | Seed for centroid initialization. |
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type | Description |
/// |----------|------|-------------|
/// | `cluster_centers()` | `MatrixType` | Coordinates of cluster centers (n_clusters × n_features). |
/// | `labels()` | `IndexVector` | Labels of each point from the best run (n_samples,). |
/// | `inertia()` | `Scalar` | Sum of squared distances of samples to their closest cluster center. |
/// | `n_iter()` | `int` | Number of iterations run in the best trial. |
///
/// ### See also
///
/// - Skigen::MiniBatchKMeans — Mini-batch variant for large datasets.
///
/// @note **scikit-learn parity gaps:** The following sklearn constructor
///   parameters are not yet supported: `init` (only k-means++), `tol`,
///   `verbose`, `copy_x`, `algorithm` (only Lloyd).
///   The following sklearn fitted attributes are not yet exposed:
///   `n_features_in_`, `feature_names_in_`.
///   The `fit_predict()`, `fit_transform()`, and `score()` methods
///   are not yet implemented.
///
/// ### Examples
///
/// @snippet kmeans.cpp example_kmeans
template <typename Scalar = double>
class KMeans {
public:
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using IndexVector = Eigen::VectorXi;

    /// @brief Construct a KMeans estimator.
    ///
    /// @param n_clusters The number of clusters (`int`, default `8`).
    /// @param max_iter Maximum iterations per run (`int`, default `300`).
    /// @param n_init Number of runs with different seeds (`int`, default `10`).
    /// @param random_state RNG seed (`unsigned int`, default `42`).
    explicit KMeans(int n_clusters = 8, int max_iter = 300,
                    int n_init = 10, unsigned int random_state = 42)
        : n_clusters_(n_clusters), max_iter_(max_iter),
          n_init_(n_init), random_state_(random_state) {}

    // -- Accessors ----------------------------------------------------------

    /// @brief Whether the estimator has been fitted.
    [[nodiscard]] bool is_fitted() const noexcept { return fitted_; }
    /// @brief The number of clusters.
    [[nodiscard]] int n_clusters() const noexcept { return n_clusters_; }

    /// @brief Cluster centers (n_clusters × n_features).
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] const MatrixType& cluster_centers() const {
        if (!fitted_) throw std::runtime_error("KMeans has not been fitted yet.");
        return cluster_centers_;
    }
    /// @brief Labels of each training point from the best run.
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] const IndexVector& labels() const {
        if (!fitted_) throw std::runtime_error("KMeans has not been fitted yet.");
        return labels_;
    }
    /// @brief Sum of squared distances to closest cluster center.
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] Scalar inertia() const {
        if (!fitted_) throw std::runtime_error("KMeans has not been fitted yet.");
        return inertia_;
    }
    /// @brief Number of iterations in the best run.
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] int n_iter() const {
        if (!fitted_) throw std::runtime_error("KMeans has not been fitted yet.");
        return n_iter_;
    }

    // -- fit / predict -------------------------------------------------------

    /// @brief Compute k-means clustering.
    ///
    /// Runs `n_init` trials of Lloyd's algorithm with k-means++
    /// initialization, keeping the result with the lowest inertia.
    ///
    /// @param X Training data of shape (n_samples, n_features).
    /// @return Reference to the fitted estimator (`*this`).
    /// @throws std::invalid_argument if `n_samples < n_clusters`.
    KMeans& fit(const Eigen::Ref<const MatrixType>& X) {
        internal::check_non_empty(X);
        if (X.rows() < n_clusters_) {
            throw std::invalid_argument(
                "n_samples must be >= n_clusters.");
        }

        Scalar best_inertia = std::numeric_limits<Scalar>::max();

        for (int init = 0; init < n_init_; ++init) {
            std::mt19937 rng(random_state_ + static_cast<unsigned>(init));

            MatrixType centers = kmeans_plus_plus(X, rng);
            IndexVector labels(X.rows());
            int iters = 0;

            for (int it = 0; it < max_iter_; ++it) {
                // Assign
                assign_labels(X, centers, labels);

                // Update
                MatrixType new_centers = MatrixType::Zero(n_clusters_, X.cols());
                Eigen::VectorXi counts = Eigen::VectorXi::Zero(n_clusters_);
                for (Eigen::Index i = 0; i < X.rows(); ++i) {
                    new_centers.row(labels(i)) += X.row(i);
                    counts(labels(i)) += 1;
                }
                for (int k = 0; k < n_clusters_; ++k) {
                    if (counts(k) > 0) {
                        new_centers.row(k) /= static_cast<Scalar>(counts(k));
                    } else {
                        // Empty cluster — reinitialize from random point
                        std::uniform_int_distribution<Eigen::Index> dist(0, X.rows() - 1);
                        new_centers.row(k) = X.row(dist(rng));
                    }
                }

                iters = it + 1;
                if ((new_centers - centers).squaredNorm() < std::numeric_limits<Scalar>::epsilon()) {
                    break;
                }
                centers = new_centers;
            }

            assign_labels(X, centers, labels);
            Scalar inertia = compute_inertia(X, centers, labels);

            if (inertia < best_inertia) {
                best_inertia = inertia;
                cluster_centers_ = centers;
                labels_ = labels;
                inertia_ = inertia;
                n_iter_ = iters;
            }
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
        if (!fitted_) throw std::runtime_error("KMeans has not been fitted yet.");
        IndexVector labels(X.rows());
        assign_labels(X, cluster_centers_, labels);
        return labels;
    }

    /// @brief Transform X to a cluster-distance space.
    ///
    /// Returns the Euclidean distance from each sample to each
    /// cluster center.
    ///
    /// @param X Data of shape (n_samples, n_features).
    /// @return Distance matrix of shape (n_samples, n_clusters).
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] MatrixType transform(const Eigen::Ref<const MatrixType>& X) const {
        if (!fitted_) throw std::runtime_error("KMeans has not been fitted yet.");
        // Returns distance to each cluster center
        MatrixType distances(X.rows(), n_clusters_);
        for (Eigen::Index i = 0; i < X.rows(); ++i) {
            for (int k = 0; k < n_clusters_; ++k) {
                distances(i, k) = (X.row(i) - cluster_centers_.row(k)).norm();
            }
        }
        return distances;
    }

private:
    int n_clusters_;
    int max_iter_;
    int n_init_;
    unsigned int random_state_;

    bool fitted_ = false;
    MatrixType cluster_centers_;
    IndexVector labels_;
    Scalar inertia_ = Scalar{0};
    int n_iter_ = 0;

    // k-means++ initialization
    MatrixType kmeans_plus_plus(const Eigen::Ref<const MatrixType>& X,
                                std::mt19937& rng) const {
        const Eigen::Index n = X.rows();
        MatrixType centers(n_clusters_, X.cols());

        // Pick first center uniformly at random
        std::uniform_int_distribution<Eigen::Index> first_dist(0, n - 1);
        centers.row(0) = X.row(first_dist(rng));

        VectorType min_dist = VectorType::Constant(n, std::numeric_limits<Scalar>::max());

        for (int k = 1; k < n_clusters_; ++k) {
            // Update min distances
            for (Eigen::Index i = 0; i < n; ++i) {
                Scalar d = (X.row(i) - centers.row(k - 1)).squaredNorm();
                min_dist(i) = std::min(min_dist(i), d);
            }

            // Sample proportional to D²
            std::discrete_distribution<Eigen::Index> dist(
                min_dist.data(), min_dist.data() + n);
            centers.row(k) = X.row(dist(rng));
        }

        return centers;
    }

    static void assign_labels(const Eigen::Ref<const MatrixType>& X,
                              const MatrixType& centers,
                              IndexVector& labels) {
        for (Eigen::Index i = 0; i < X.rows(); ++i) {
            Scalar best_dist = std::numeric_limits<Scalar>::max();
            int best_k = 0;
            for (int k = 0; k < centers.rows(); ++k) {
                Scalar d = (X.row(i) - centers.row(k)).squaredNorm();
                if (d < best_dist) {
                    best_dist = d;
                    best_k = k;
                }
            }
            labels(i) = best_k;
        }
    }

    static Scalar compute_inertia(const Eigen::Ref<const MatrixType>& X,
                                  const MatrixType& centers,
                                  const IndexVector& labels) {
        Scalar inertia{0};
        for (Eigen::Index i = 0; i < X.rows(); ++i) {
            inertia += (X.row(i) - centers.row(labels(i))).squaredNorm();
        }
        return inertia;
    }
};

/// @}

} // namespace Skigen

#endif // SKIGEN_CLUSTER_KMEANS_H
