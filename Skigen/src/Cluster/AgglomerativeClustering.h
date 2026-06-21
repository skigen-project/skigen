// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

#ifndef SKIGEN_CLUSTER_AGGLOMERATIVE_CLUSTERING_H
#define SKIGEN_CLUSTER_AGGLOMERATIVE_CLUSTERING_H

#include "../Core/Base.h"
#include "../Core/Validation.h"

#include <Eigen/Core>
#include <algorithm>
#include <cstddef>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace Skigen {

/// @defgroup Algo_AgglomerativeClustering AgglomerativeClustering
/// @ingroup Cluster
/// @brief Hierarchical agglomerative clustering.
/// @{

/// @brief Bottom-up hierarchical clustering with sklearn-style labels.
///
/// Starts with one cluster per sample and repeatedly merges the closest pair.
/// The full merge tree is stored in `children()`, and `labels()` are taken by
/// cutting the tree at `n_clusters` active clusters.
///
/// Mirrors the dense, brute-force subset of
/// [sklearn.cluster.AgglomerativeClustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html).
///
/// ### Examples
///
/// @snippet agglomerative_clustering.cpp example_agglomerative_clustering
template <typename Scalar = double>
class AgglomerativeClustering : public Estimator<AgglomerativeClustering<Scalar>, Scalar> {
public:
    using Base = Estimator<AgglomerativeClustering<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using IndexVector = Eigen::VectorXi;

    /// @brief Construct an AgglomerativeClustering estimator.
    ///
    /// @param n_clusters Number of clusters to find (`int`, default `2`).
    /// @param linkage Linkage criterion: `ward`, `single`, `complete`, or `average`.
    /// @param metric Distance metric for non-Ward linkage: `euclidean` or `manhattan`.
    explicit AgglomerativeClustering(int n_clusters = 2, std::string linkage = "ward", std::string metric = "euclidean")
        : n_clusters_(n_clusters), linkage_(std::move(linkage)), metric_(std::move(metric)) {}

    /// @brief Number of output clusters.
    [[nodiscard]] int n_clusters() const noexcept { return n_clusters_; }

    /// @brief Linkage criterion.
    [[nodiscard]] const std::string& linkage() const noexcept { return linkage_; }

    /// @brief Distance metric.
    [[nodiscard]] const std::string& metric() const noexcept { return metric_; }

    /// @brief Cluster label for each training sample.
    [[nodiscard]] const IndexVector& labels() const {
        this->check_is_fitted();
        return labels_;
    }

    /// @brief Merge tree children, shape `(n_samples - 1, 2)`.
    [[nodiscard]] const Eigen::MatrixXi& children() const {
        this->check_is_fitted();
        return children_;
    }

    /// @brief Distance or Ward cost associated with each merge.
    [[nodiscard]] const VectorType& distances() const {
        this->check_is_fitted();
        return distances_;
    }

    SKIGEN_PARAMS(
        (n_clusters, n_clusters_, int),
        (linkage, linkage_, std::string),
        (metric, metric_, std::string))

    /// @brief Fit the hierarchical clustering tree.
    AgglomerativeClustering& fit(const Eigen::Ref<const MatrixType>& X) {
        internal::check_non_empty(X);
        validate_parameters(X.rows());

        const Eigen::Index sample_count = X.rows();
        std::vector<ClusterNode> active;
        active.reserve(static_cast<std::size_t>(sample_count));
        for (Eigen::Index sample_index = 0; sample_index < sample_count; ++sample_index) {
            active.push_back(ClusterNode{static_cast<int>(sample_index), {static_cast<int>(sample_index)}});
        }

        children_.resize(sample_count - 1, 2);
        distances_.resize(sample_count - 1);
        labels_.resize(sample_count);
        if (n_clusters_ == sample_count) assign_labels(active);

        for (Eigen::Index merge_index = 0; merge_index < sample_count - 1; ++merge_index) {
            Eigen::Index best_left = 0;
            Eigen::Index best_right = 1;
            Scalar best_distance = std::numeric_limits<Scalar>::max();

            for (Eigen::Index left_index = 0; left_index < static_cast<Eigen::Index>(active.size()); ++left_index) {
                for (Eigen::Index right_index = left_index + 1; right_index < static_cast<Eigen::Index>(active.size()); ++right_index) {
                    const Scalar distance = cluster_distance(X, active[static_cast<std::size_t>(left_index)],
                                                             active[static_cast<std::size_t>(right_index)]);
                    if (distance < best_distance) {
                        best_distance = distance;
                        best_left = left_index;
                        best_right = right_index;
                    }
                }
            }

            ClusterNode merged;
            merged.id = static_cast<int>(sample_count + merge_index);
            const auto left_slot = static_cast<std::size_t>(best_left);
            const auto right_slot = static_cast<std::size_t>(best_right);
            children_(merge_index, 0) = active[left_slot].id;
            children_(merge_index, 1) = active[right_slot].id;
            distances_(merge_index) = best_distance;
            merged.members = active[left_slot].members;
            merged.members.insert(merged.members.end(), active[right_slot].members.begin(), active[right_slot].members.end());
            std::sort(merged.members.begin(), merged.members.end());

            active.erase(active.begin() + static_cast<std::ptrdiff_t>(best_right));
            active.erase(active.begin() + static_cast<std::ptrdiff_t>(best_left));
            active.push_back(std::move(merged));

            if (static_cast<int>(active.size()) == n_clusters_) assign_labels(active);
        }
        if (n_clusters_ == 1) labels_.setZero();

        this->n_features_in_ = X.cols();
        this->fitted_ = true;
        return *this;
    }

    /// @brief Fit and return training labels.
    [[nodiscard]] IndexVector fit_predict(const Eigen::Ref<const MatrixType>& X) {
        fit(X);
        return labels_;
    }

private:
    struct ClusterNode {
        int id;
        std::vector<int> members;
    };

    int n_clusters_;
    std::string linkage_;
    std::string metric_;
    IndexVector labels_;
    Eigen::MatrixXi children_;
    VectorType distances_;

    void validate_parameters(Eigen::Index sample_count) const {
        if (n_clusters_ <= 0) {
            throw std::invalid_argument("AgglomerativeClustering: n_clusters must be positive.");
        }
        if (n_clusters_ > sample_count) {
            throw std::invalid_argument("AgglomerativeClustering: n_clusters must be <= n_samples.");
        }
        if (!is_supported_linkage(linkage_)) {
            throw std::invalid_argument("AgglomerativeClustering: linkage must be 'ward', 'single', 'complete', or 'average'.");
        }
        if (!is_supported_metric(metric_)) {
            throw std::invalid_argument("AgglomerativeClustering: metric must be 'euclidean' or 'manhattan'.");
        }
        if (linkage_ == "ward" && !(metric_ == "euclidean" || metric_ == "l2")) {
            throw std::invalid_argument("AgglomerativeClustering: ward linkage requires euclidean metric.");
        }
    }

    static bool is_supported_linkage(const std::string& linkage) {
        return linkage == "ward" || linkage == "single" || linkage == "complete" || linkage == "average";
    }

    static bool is_supported_metric(const std::string& metric) {
        return metric == "euclidean" || metric == "l2" || metric == "manhattan" || metric == "l1";
    }

    Scalar point_distance(const Eigen::Ref<const MatrixType>& X, int first_index, int second_index) const {
        if (metric_ == "manhattan" || metric_ == "l1") {
            return (X.row(first_index) - X.row(second_index)).array().abs().sum();
        }
        return (X.row(first_index) - X.row(second_index)).norm();
    }

    MatrixType cluster_centroid(const Eigen::Ref<const MatrixType>& X, const ClusterNode& cluster) const {
        MatrixType centroid = MatrixType::Zero(1, X.cols());
        for (const int member_index : cluster.members) centroid.row(0) += X.row(member_index);
        centroid /= static_cast<Scalar>(cluster.members.size());
        return centroid;
    }

    Scalar cluster_distance(const Eigen::Ref<const MatrixType>& X,
                            const ClusterNode& left,
                            const ClusterNode& right) const {
        if (linkage_ == "ward") {
            const MatrixType left_centroid = cluster_centroid(X, left);
            const MatrixType right_centroid = cluster_centroid(X, right);
            const Scalar left_size = static_cast<Scalar>(left.members.size());
            const Scalar right_size = static_cast<Scalar>(right.members.size());
            return (left_size * right_size / (left_size + right_size)) *
                   (left_centroid.row(0) - right_centroid.row(0)).squaredNorm();
        }

        if (linkage_ == "single") {
            Scalar best = std::numeric_limits<Scalar>::max();
            for (const int left_member : left.members) {
                for (const int right_member : right.members) {
                    best = std::min(best, point_distance(X, left_member, right_member));
                }
            }
            return best;
        }

        if (linkage_ == "complete") {
            Scalar worst = Scalar{0};
            for (const int left_member : left.members) {
                for (const int right_member : right.members) {
                    worst = std::max(worst, point_distance(X, left_member, right_member));
                }
            }
            return worst;
        }

        Scalar total = Scalar{0};
        Eigen::Index pair_count = 0;
        for (const int left_member : left.members) {
            for (const int right_member : right.members) {
                total += point_distance(X, left_member, right_member);
                ++pair_count;
            }
        }
        return total / static_cast<Scalar>(pair_count);
    }

    void assign_labels(const std::vector<ClusterNode>& active) {
        for (Eigen::Index cluster_index = 0; cluster_index < static_cast<Eigen::Index>(active.size()); ++cluster_index) {
            for (const int member_index : active[static_cast<std::size_t>(cluster_index)].members) {
                labels_(member_index) = static_cast<int>(cluster_index);
            }
        }
    }
};

/// @}

} // namespace Skigen

#endif // SKIGEN_CLUSTER_AGGLOMERATIVE_CLUSTERING_H