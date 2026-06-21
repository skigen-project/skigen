// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

#ifndef SKIGEN_CLUSTER_DBSCAN_H
#define SKIGEN_CLUSTER_DBSCAN_H

#include "../Core/Base.h"
#include "../Core/Validation.h"
#include "../Neighbors/KNeighbors.h"

#include <Eigen/Core>
#include <algorithm>
#include <cstddef>
#include <deque>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace Skigen {

/// @defgroup Algo_DBSCAN DBSCAN
/// @ingroup Cluster
/// @brief Density-based clustering with noise labels.
/// @{

/// @brief Density-Based Spatial Clustering of Applications with Noise.
///
/// DBSCAN groups samples that are density-connected through core samples.
/// Samples not reachable from any core component are labelled `-1`, matching
/// scikit-learn's noise convention.
///
/// Mirrors the dense, brute-force subset of
/// [sklearn.cluster.DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html).
///
/// ### Examples
///
/// @snippet dbscan.cpp example_dbscan
template <typename Scalar = double>
class DBSCAN : public Estimator<DBSCAN<Scalar>, Scalar> {
public:
    using Base = Estimator<DBSCAN<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::IndexType;
    using IndexVector = Eigen::VectorXi;

    /// @brief Construct a DBSCAN estimator.
    ///
    /// @param eps Maximum neighbourhood radius (`Scalar`, default `0.5`).
    /// @param min_samples Minimum samples in an eps-neighbourhood for a core point (`int`, default `5`).
    /// @param metric Distance metric: `euclidean` or `manhattan` (`std::string`, default `euclidean`).
    explicit DBSCAN(Scalar eps = Scalar{0.5}, int min_samples = 5, std::string metric = "euclidean")
        : eps_(eps), min_samples_(min_samples), metric_(std::move(metric)) {}

    /// @brief Maximum neighbourhood radius.
    [[nodiscard]] Scalar eps() const noexcept { return eps_; }

    /// @brief Minimum samples required for a core point.
    [[nodiscard]] int min_samples() const noexcept { return min_samples_; }

    /// @brief Distance metric name.
    [[nodiscard]] const std::string& metric() const noexcept { return metric_; }

    /// @brief Cluster label for each training sample (`-1` marks noise).
    [[nodiscard]] const IndexVector& labels() const {
        this->check_is_fitted();
        return labels_;
    }

    /// @brief Indices of core samples in the training data.
    [[nodiscard]] const IndexVector& core_sample_indices() const {
        this->check_is_fitted();
        return core_sample_indices_;
    }

    /// @brief Dense rows corresponding to `core_sample_indices()`.
    [[nodiscard]] const MatrixType& components() const {
        this->check_is_fitted();
        return components_;
    }

    SKIGEN_PARAMS(
        (eps, eps_, double),
        (min_samples, min_samples_, int),
        (metric, metric_, std::string))

    /// @brief Fit DBSCAN from a dense design matrix.
    ///
    /// @param X Training data of shape (n_samples, n_features).
    /// @return Reference to the fitted estimator (`*this`).
    DBSCAN& fit(const Eigen::Ref<const MatrixType>& X) {
        internal::check_non_empty(X);
        validate_parameters();

        const Eigen::Index sample_count = X.rows();
        auto neighbourhoods = radius_neighbourhoods(X);

        std::vector<bool> visited(static_cast<std::size_t>(sample_count), false);
        labels_ = IndexVector::Constant(sample_count, -1);
        std::vector<int> core_indices;
        core_indices.reserve(static_cast<std::size_t>(sample_count));

        std::vector<bool> is_core(static_cast<std::size_t>(sample_count), false);
        for (Eigen::Index sample_index = 0; sample_index < sample_count; ++sample_index) {
            const bool core = static_cast<int>(neighbourhoods[static_cast<std::size_t>(sample_index)].size()) >= min_samples_;
            is_core[static_cast<std::size_t>(sample_index)] = core;
            if (core) core_indices.push_back(static_cast<int>(sample_index));
        }

        int cluster_id = 0;
        for (Eigen::Index sample_index = 0; sample_index < sample_count; ++sample_index) {
            const auto sample_slot = static_cast<std::size_t>(sample_index);
            if (visited[sample_slot]) continue;
            visited[sample_slot] = true;

            if (!is_core[sample_slot]) {
                labels_(sample_index) = -1;
                continue;
            }

            labels_(sample_index) = cluster_id;
            std::deque<int> seeds;
            for (const int neighbour_index : neighbourhoods[sample_slot]) {
                if (neighbour_index != sample_index) seeds.push_back(neighbour_index);
            }

            while (!seeds.empty()) {
                const int current_index = seeds.front();
                seeds.pop_front();
                const auto current_slot = static_cast<std::size_t>(current_index);

                if (!visited[current_slot]) {
                    visited[current_slot] = true;
                    if (is_core[current_slot]) {
                        for (const int next_index : neighbourhoods[current_slot]) {
                            if (!visited[static_cast<std::size_t>(next_index)]) {
                                seeds.push_back(next_index);
                            }
                        }
                    }
                }
                if (labels_(current_index) == -1) labels_(current_index) = cluster_id;
            }
            ++cluster_id;
        }

        core_sample_indices_ = to_index_vector(core_indices);
        components_.resize(static_cast<Eigen::Index>(core_indices.size()), X.cols());
        for (Eigen::Index row_index = 0; row_index < components_.rows(); ++row_index) {
            components_.row(row_index) = X.row(core_sample_indices_(row_index));
        }

        this->n_features_in_ = X.cols();
        this->fitted_ = true;
        return *this;
    }

    /// @brief Fit DBSCAN and return training labels.
    [[nodiscard]] IndexVector fit_predict(const Eigen::Ref<const MatrixType>& X) {
        fit(X);
        return labels_;
    }

private:
    Scalar eps_;
    int min_samples_;
    std::string metric_;
    IndexVector labels_;
    IndexVector core_sample_indices_;
    MatrixType components_;

    void validate_parameters() const {
        if (eps_ <= Scalar{0}) {
            throw std::invalid_argument("DBSCAN: eps must be positive.");
        }
        if (min_samples_ <= 0) {
            throw std::invalid_argument("DBSCAN: min_samples must be positive.");
        }
        if (!is_supported_metric(metric_)) {
            throw std::invalid_argument("DBSCAN: metric must be 'euclidean' or 'manhattan'.");
        }
    }

    static bool is_supported_metric(const std::string& metric) {
        return metric == "euclidean" || metric == "l2" || metric == "manhattan" || metric == "l1";
    }

    std::vector<std::vector<int>> radius_neighbourhoods(const Eigen::Ref<const MatrixType>& X) const {
        if (metric_ == "euclidean" || metric_ == "l2") {
            const Scalar eps_squared = eps_ * eps_;
            const auto pairs = internal::sorted_neighbor_pairs<Scalar>(X, X);
            std::vector<std::vector<int>> neighbourhoods;
            neighbourhoods.reserve(static_cast<std::size_t>(X.rows()));
            for (const auto& row_pairs : pairs) {
                std::vector<int> row;
                for (const auto& distance_index : row_pairs) {
                    if (distance_index.first <= eps_squared) row.push_back(distance_index.second);
                }
                neighbourhoods.push_back(std::move(row));
            }
            return neighbourhoods;
        }

        std::vector<std::vector<int>> neighbourhoods;
        neighbourhoods.reserve(static_cast<std::size_t>(X.rows()));
        for (Eigen::Index sample_index = 0; sample_index < X.rows(); ++sample_index) {
            std::vector<int> row;
            for (Eigen::Index candidate_index = 0; candidate_index < X.rows(); ++candidate_index) {
                if ((X.row(sample_index) - X.row(candidate_index)).array().abs().sum() <= eps_) {
                    row.push_back(static_cast<int>(candidate_index));
                }
            }
            neighbourhoods.push_back(std::move(row));
        }
        return neighbourhoods;
    }

    static IndexVector to_index_vector(const std::vector<int>& indices) {
        IndexVector out(static_cast<Eigen::Index>(indices.size()));
        for (Eigen::Index index = 0; index < out.size(); ++index) {
            out(index) = indices[static_cast<std::size_t>(index)];
        }
        return out;
    }
};

/// @}

} // namespace Skigen

#endif // SKIGEN_CLUSTER_DBSCAN_H