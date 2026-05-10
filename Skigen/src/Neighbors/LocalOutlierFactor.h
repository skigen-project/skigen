// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_NEIGHBORS_LOCAL_OUTLIER_FACTOR_H
#define SKIGEN_NEIGHBORS_LOCAL_OUTLIER_FACTOR_H

#include "../Core/Base.h"
#include "../Core/Validation.h"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace Skigen {

/// @defgroup Algo_LocalOutlierFactor Local Outlier Factor
/// @ingroup Neighbors
/// @brief Unsupervised outlier detection using Local Outlier Factor (LOF).
/// @{

/// @brief Unsupervised Outlier Detection using the Local Outlier Factor.
///
/// The anomaly score of each sample is called the Local Outlier Factor.
/// It measures the local deviation of the density of a given sample
/// with respect to its neighbors. It is local in that the anomaly
/// score depends on how isolated the object is with respect to the
/// surrounding neighborhood. More precisely, locality is given by
/// k-nearest neighbors, whose distance is used to estimate the local
/// density.
///
/// By comparing the local density of a sample to the local densities
/// of its neighbors, one can identify samples that have a substantially
/// lower density than their neighbors. These are considered outliers.
///
/// Mirrors
/// [sklearn.neighbors.LocalOutlierFactor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html).
///
/// ### Algorithm
///
/// For each sample @f$x_i@f$:
///
/// 1. Compute the pairwise Euclidean distance to all other samples.
/// 2. Find the @f$k@f$ nearest neighbors @f$N_k(x_i)@f$.
/// 3. Compute the @f$k@f$-distance: @f$d_k(x_i)@f$ = distance to
///    the @f$k@f$-th nearest neighbor.
/// 4. Compute the reachability distance:
///    @f$\mathrm{reach\_dist}_k(x_i, x_j) = \max\bigl(d_k(x_j),\; d(x_i, x_j)\bigr)@f$
/// 5. Compute the local reachability density:
///    @f$\mathrm{lrd}_k(x_i) = \frac{k}{\sum_{x_j \in N_k(x_i)} \mathrm{reach\_dist}_k(x_i, x_j)}@f$
/// 6. Compute the LOF score:
///    @f$\mathrm{LOF}_k(x_i) = \frac{1}{k} \sum_{x_j \in N_k(x_i)} \frac{\mathrm{lrd}_k(x_j)}{\mathrm{lrd}_k(x_i)}@f$
///
/// LOF ≈ 1 means similar density to neighbors (inlier).
/// LOF >> 1 means much lower density (outlier).
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `n_neighbors` | `int` | `20` | Number of neighbors to use. |
/// | `contamination` | `Scalar` | `-1` | Proportion of outliers (for threshold). `-1` means no automatic threshold; use `negative_outlier_factor()` directly. |
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type | Description |
/// |----------|------|-------------|
/// | `negative_outlier_factor()` | `VectorType` | Negative LOF scores (n_samples,). Lower = more anomalous. |
/// | `n_neighbors_used()` | `int` | Effective number of neighbors (clamped to n_samples − 1). |
/// | `offset()` | `Scalar` | Threshold for outlier labeling (−1.5 if contamination not set). |
///
/// ### Notes
///
/// This implementation uses brute-force pairwise Euclidean distances,
/// giving @f$O(n^2 m + n^2 \log k)@f$ time complexity where @f$n@f$ is
/// the number of samples and @f$m@f$ is the number of features.
///
/// ### Limitations relative to scikit-learn `algorithm`, `leaf_size`,
///   `metric`, `metric_params`, `p`, `n_jobs`, `novelty` are not yet
///   supported.
///
/// ### Examples
///
/// @snippet local_outlier_factor.cpp example_lof
template <typename Scalar = double>
class LocalOutlierFactor
    : public Estimator<LocalOutlierFactor<Scalar>, Scalar> {
public:
    using Base = Estimator<LocalOutlierFactor<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::IndexType;
    using typename Base::ScalarType;

    /// @brief Construct a LocalOutlierFactor estimator.
    ///
    /// @param n_neighbors Number of neighbors to use (`int`, default `20`).
    /// @param contamination Proportion of outliers (`Scalar`, default `-1`
    ///   meaning no automatic threshold).
    explicit LocalOutlierFactor(int n_neighbors = 20,
                                Scalar contamination = Scalar{-1})
        : n_neighbors_(n_neighbors), contamination_(contamination) {}

    // -- Accessors ----------------------------------------------------------

    /// @brief Number of neighbors requested.
    [[nodiscard]] int n_neighbors() const noexcept { return n_neighbors_; }

    /// @brief Effective number of neighbors used (clamped to n_samples − 1).
    [[nodiscard]] int n_neighbors_used() const {
        this->check_is_fitted();
        return n_neighbors_used_;
    }

    /// @brief Negative LOF scores of the training samples (n_samples,).
    ///
    /// The negated LOF scores. More negative = more outlying.
    /// Inliers tend to have scores close to −1.
    [[nodiscard]] const VectorType& negative_outlier_factor() const {
        this->check_is_fitted();
        return negative_outlier_factor_;
    }

    /// @brief Threshold for labeling outliers.
    ///
    /// If contamination was set, this is the score percentile threshold.
    /// Otherwise defaults to −1.5.
    [[nodiscard]] Scalar offset() const {
        this->check_is_fitted();
        return offset_;
    }

    // -- fit / predict -------------------------------------------------------

    /// @brief Fit the LOF model from the training data.
    ///
    /// Computes the LOF score for every training sample.
    ///
    /// @param X Training data of shape (n_samples, n_features).
    /// @return Reference to the fitted estimator (`*this`).
    /// @throws std::invalid_argument if n_samples < 2.
    LocalOutlierFactor& fit(const Eigen::Ref<const MatrixType>& X) {
        internal::check_non_empty(X);
        const auto n = X.rows();

        if (n < 2) {
            throw std::invalid_argument(
                "LocalOutlierFactor requires at least 2 samples, "
                "got " + std::to_string(n) + ".");
        }

        this->n_features_in_ = X.cols();
        n_neighbors_used_ = std::min(n_neighbors_, static_cast<int>(n - 1));
        const int k = n_neighbors_used_;

        // 1. Pairwise Euclidean distance matrix (symmetric)
        MatrixType dist = MatrixType::Zero(n, n);
        for (Eigen::Index i = 0; i < n; ++i) {
            for (Eigen::Index j = i + 1; j < n; ++j) {
                Scalar d = (X.row(i) - X.row(j)).norm();
                dist(i, j) = d;
                dist(j, i) = d;
            }
        }

        // 2. k-nearest neighbors and k-distance for each point
        std::vector<std::vector<int>> neighbors(static_cast<std::size_t>(n));
        VectorType k_dist(n);

        for (Eigen::Index i = 0; i < n; ++i) {
            std::vector<int> indices(static_cast<std::size_t>(n));
            std::iota(indices.begin(), indices.end(), 0);
            std::sort(indices.begin(), indices.end(),
                      [&dist, i](int a, int b) {
                          return dist(i, a) < dist(i, b);
                      });

            // Skip self (index 0 after sorting), take next k
            auto& nb = neighbors[static_cast<std::size_t>(i)];
            nb.resize(static_cast<std::size_t>(k));
            for (int j = 0; j < k; ++j) {
                nb[static_cast<std::size_t>(j)] =
                    indices[static_cast<std::size_t>(j + 1)];
            }
            k_dist(i) = dist(i, indices[static_cast<std::size_t>(k)]);
        }

        // 3. Local Reachability Density (LRD)
        VectorType lrd(n);
        constexpr Scalar eps = Scalar{1e-30};
        for (Eigen::Index i = 0; i < n; ++i) {
            Scalar reach_sum{0};
            for (int j = 0; j < k; ++j) {
                int nb = neighbors[static_cast<std::size_t>(i)]
                                  [static_cast<std::size_t>(j)];
                reach_sum += std::max(k_dist(nb), dist(i, nb));
            }
            lrd(i) = (reach_sum > eps)
                ? static_cast<Scalar>(k) / reach_sum
                : Scalar{1};
        }

        // 4. LOF scores
        VectorType lof(n);
        for (Eigen::Index i = 0; i < n; ++i) {
            Scalar lrd_sum{0};
            for (int j = 0; j < k; ++j) {
                int nb = neighbors[static_cast<std::size_t>(i)]
                                  [static_cast<std::size_t>(j)];
                lrd_sum += lrd(nb);
            }
            Scalar avg_lrd = lrd_sum / static_cast<Scalar>(k);
            lof(i) = (lrd(i) > eps) ? avg_lrd / lrd(i) : Scalar{1};
        }

        // Store as negative outlier factor (sklearn convention)
        negative_outlier_factor_ = -lof;

        // Compute offset
        if (contamination_ > Scalar{0} && contamination_ < Scalar{1}) {
            // Percentile threshold
            std::vector<Scalar> sorted_scores(
                negative_outlier_factor_.data(),
                negative_outlier_factor_.data() + n);
            std::sort(sorted_scores.begin(), sorted_scores.end());
            auto idx = static_cast<std::size_t>(
                std::floor(contamination_ * static_cast<Scalar>(n)));
            if (idx >= sorted_scores.size()) idx = sorted_scores.size() - 1;
            offset_ = sorted_scores[idx];
        } else {
            offset_ = Scalar{-1.5};
        }

        this->fitted_ = true;
        return *this;
    }

    /// @brief Predict outlier labels for the training data.
    ///
    /// Returns +1 for inliers and −1 for outliers, based on the
    /// `negative_outlier_factor()` and the `offset()`.
    ///
    /// @return Label vector of shape (n_samples,). +1 = inlier, −1 = outlier.
    [[nodiscard]] Eigen::VectorXi fit_predict_labels() const {
        this->check_is_fitted();
        const auto n = negative_outlier_factor_.size();
        Eigen::VectorXi labels(n);
        for (Eigen::Index i = 0; i < n; ++i) {
            labels(i) = (negative_outlier_factor_(i) >= offset_) ? 1 : -1;
        }
        return labels;
    }

    /// @brief Return the raw (positive) LOF scores for the training data.
    ///
    /// Convenience accessor: returns `−negative_outlier_factor()`.
    ///
    /// @return LOF score vector of shape (n_samples,).
    [[nodiscard]] VectorType lof_scores() const {
        this->check_is_fitted();
        return -negative_outlier_factor_;
    }

private:
    int n_neighbors_;
    Scalar contamination_;

    int n_neighbors_used_ = 0;
    VectorType negative_outlier_factor_;
    Scalar offset_{0};
};

/// @}

} // namespace Skigen

#endif // SKIGEN_NEIGHBORS_LOCAL_OUTLIER_FACTOR_H
