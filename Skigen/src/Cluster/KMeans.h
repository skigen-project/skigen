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

/// KMeans — K-Means clustering via Lloyd's algorithm with k-means++ init.
/// Mirrors sklearn.cluster.KMeans.
template <typename Scalar = double>
class KMeans {
public:
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using IndexVector = Eigen::VectorXi;

    explicit KMeans(int n_clusters = 8, int max_iter = 300,
                    int n_init = 10, unsigned int random_state = 42)
        : n_clusters_(n_clusters), max_iter_(max_iter),
          n_init_(n_init), random_state_(random_state) {}

    // -- Accessors ----------------------------------------------------------

    [[nodiscard]] bool is_fitted() const noexcept { return fitted_; }
    [[nodiscard]] int n_clusters() const noexcept { return n_clusters_; }

    [[nodiscard]] const MatrixType& cluster_centers() const {
        if (!fitted_) throw std::runtime_error("KMeans has not been fitted yet.");
        return cluster_centers_;
    }
    [[nodiscard]] const IndexVector& labels() const {
        if (!fitted_) throw std::runtime_error("KMeans has not been fitted yet.");
        return labels_;
    }
    [[nodiscard]] Scalar inertia() const {
        if (!fitted_) throw std::runtime_error("KMeans has not been fitted yet.");
        return inertia_;
    }
    [[nodiscard]] int n_iter() const {
        if (!fitted_) throw std::runtime_error("KMeans has not been fitted yet.");
        return n_iter_;
    }

    // -- fit / predict -------------------------------------------------------

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

    [[nodiscard]] IndexVector predict(const Eigen::Ref<const MatrixType>& X) const {
        if (!fitted_) throw std::runtime_error("KMeans has not been fitted yet.");
        IndexVector labels(X.rows());
        assign_labels(X, cluster_centers_, labels);
        return labels;
    }

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

} // namespace Skigen

#endif // SKIGEN_CLUSTER_KMEANS_H
