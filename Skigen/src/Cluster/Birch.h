// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

#ifndef SKIGEN_CLUSTER_BIRCH_H
#define SKIGEN_CLUSTER_BIRCH_H

#include "../Core/Base.h"
#include "../Core/Validation.h"
#include "KMeans.h"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace Skigen {

/// @defgroup Algo_Birch Birch
/// @ingroup Cluster
/// @brief Balanced iterative reducing and clustering using hierarchies.
/// @{

/// @brief Dense BIRCH-style clustering with flat CF subclusters.
///
/// This first Skigen implementation compresses the input into clustering
/// feature (CF) subclusters using `threshold`, then globally clusters the
/// subcluster centroids to `n_clusters` centers. It keeps the public estimator
/// shape close to sklearn while staying dependency-free and compact.
///
/// Mirrors the dense core of
/// [sklearn.cluster.Birch](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html).
///
/// ### Examples
///
/// @snippet birch.cpp example_birch
template <typename Scalar = double>
class Birch : public Estimator<Birch<Scalar>, Scalar> {
public:
    using Base = Estimator<Birch<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::RowVectorType;
    using IndexVector = Eigen::VectorXi;

    /// @brief Construct a Birch estimator.
    ///
    /// @param threshold Maximum CF subcluster radius (`Scalar`, default `0.5`).
    /// @param n_clusters Number of final clusters (`int`, default `3`).
    explicit Birch(Scalar threshold = Scalar{0.5}, int n_clusters = 3)
        : threshold_(threshold), n_clusters_(n_clusters) {}

    /// @brief Maximum CF subcluster radius.
    [[nodiscard]] Scalar threshold() const noexcept { return threshold_; }

    /// @brief Number of final clusters.
    [[nodiscard]] int n_clusters() const noexcept { return n_clusters_; }

    /// @brief Final cluster centers, shape `(n_clusters, n_features)`.
    [[nodiscard]] const MatrixType& cluster_centers() const {
        this->check_is_fitted();
        return cluster_centers_;
    }

    /// @brief CF subcluster centers produced during compression.
    [[nodiscard]] const MatrixType& subcluster_centers() const {
        this->check_is_fitted();
        return subcluster_centers_;
    }

    /// @brief Label for each training sample.
    [[nodiscard]] const IndexVector& labels() const {
        this->check_is_fitted();
        return labels_;
    }

    SKIGEN_PARAMS(
        (threshold, threshold_, double),
        (n_clusters, n_clusters_, int))

    /// @brief Fit BIRCH-style CF compression and global clustering.
    Birch& fit(const Eigen::Ref<const MatrixType>& X) {
        internal::check_non_empty(X);
        validate_parameters();

        std::vector<Subcluster> subclusters;
        subclusters.reserve(static_cast<std::size_t>(X.rows()));

        for (Eigen::Index sample_index = 0; sample_index < X.rows(); ++sample_index) {
            const RowVectorType sample = X.row(sample_index);
            if (subclusters.empty()) {
                subclusters.push_back(make_subcluster(sample));
                continue;
            }

            const int nearest = nearest_subcluster(sample, subclusters);
            if (radius_squared_after_add(subclusters[static_cast<std::size_t>(nearest)], sample) <=
                threshold_ * threshold_) {
                add_sample(subclusters[static_cast<std::size_t>(nearest)], sample);
            } else {
                subclusters.push_back(make_subcluster(sample));
            }
        }

        subcluster_centers_ = centers_from_subclusters(subclusters, X.cols());
        if (subcluster_centers_.rows() <= n_clusters_) {
            cluster_centers_ = subcluster_centers_;
        } else {
            KMeans<Scalar> kmeans(n_clusters_, /*max_iter=*/100, /*n_init=*/5, /*random_state=*/0);
            kmeans.fit(subcluster_centers_);
            cluster_centers_ = kmeans.cluster_centers();
        }

        labels_ = assign_labels(X, cluster_centers_);
        this->n_features_in_ = X.cols();
        this->fitted_ = true;
        return *this;
    }

    /// @brief Fit and return labels for the training samples.
    [[nodiscard]] IndexVector fit_predict(const Eigen::Ref<const MatrixType>& X) {
        fit(X);
        return labels_;
    }

    /// @brief Predict nearest final Birch cluster for new samples.
    [[nodiscard]] IndexVector predict(const Eigen::Ref<const MatrixType>& X) const {
        this->check_is_fitted();
        this->validate_feature_count(X);
        return assign_labels(X, cluster_centers_);
    }

private:
    struct Subcluster {
        RowVectorType linear_sum;
        Scalar squared_sum;
        int count;
    };

    Scalar threshold_;
    int n_clusters_;
    MatrixType cluster_centers_;
    MatrixType subcluster_centers_;
    IndexVector labels_;

    void validate_parameters() const {
        if (threshold_ <= Scalar{0}) {
            throw std::invalid_argument("Birch: threshold must be positive.");
        }
        if (n_clusters_ <= 0) {
            throw std::invalid_argument("Birch: n_clusters must be positive.");
        }
    }

    static Subcluster make_subcluster(const RowVectorType& sample) {
        return Subcluster{sample, sample.squaredNorm(), 1};
    }

    static RowVectorType center(const Subcluster& subcluster) {
        return subcluster.linear_sum / static_cast<Scalar>(subcluster.count);
    }

    static void add_sample(Subcluster& subcluster, const RowVectorType& sample) {
        subcluster.linear_sum += sample;
        subcluster.squared_sum += sample.squaredNorm();
        subcluster.count += 1;
    }

    static Scalar radius_squared_after_add(const Subcluster& subcluster, const RowVectorType& sample) {
        const RowVectorType linear_sum = subcluster.linear_sum + sample;
        const Scalar squared_sum = subcluster.squared_sum + sample.squaredNorm();
        const Scalar count = static_cast<Scalar>(subcluster.count + 1);
        const RowVectorType centroid = linear_sum / count;
        return std::max(Scalar{0}, squared_sum / count - centroid.squaredNorm());
    }

    static int nearest_subcluster(const RowVectorType& sample, const std::vector<Subcluster>& subclusters) {
        Scalar best_distance = std::numeric_limits<Scalar>::max();
        int best_index = 0;
        for (Eigen::Index index = 0; index < static_cast<Eigen::Index>(subclusters.size()); ++index) {
            const Scalar distance = (sample - center(subclusters[static_cast<std::size_t>(index)])).squaredNorm();
            if (distance < best_distance) {
                best_distance = distance;
                best_index = static_cast<int>(index);
            }
        }
        return best_index;
    }

    static MatrixType centers_from_subclusters(const std::vector<Subcluster>& subclusters, Eigen::Index feature_count) {
        MatrixType centers(static_cast<Eigen::Index>(subclusters.size()), feature_count);
        for (Eigen::Index index = 0; index < centers.rows(); ++index) {
            centers.row(index) = center(subclusters[static_cast<std::size_t>(index)]);
        }
        return centers;
    }

    static IndexVector assign_labels(const Eigen::Ref<const MatrixType>& X, const MatrixType& centers) {
        IndexVector labels(X.rows());
        for (Eigen::Index sample_index = 0; sample_index < X.rows(); ++sample_index) {
            Scalar best_distance = std::numeric_limits<Scalar>::max();
            int best_label = 0;
            for (Eigen::Index center_index = 0; center_index < centers.rows(); ++center_index) {
                const Scalar distance = (X.row(sample_index) - centers.row(center_index)).squaredNorm();
                if (distance < best_distance) {
                    best_distance = distance;
                    best_label = static_cast<int>(center_index);
                }
            }
            labels(sample_index) = best_label;
        }
        return labels;
    }
};

/// @}

} // namespace Skigen

#endif // SKIGEN_CLUSTER_BIRCH_H