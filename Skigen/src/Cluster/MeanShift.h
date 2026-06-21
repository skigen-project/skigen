// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

#ifndef SKIGEN_CLUSTER_MEAN_SHIFT_H
#define SKIGEN_CLUSTER_MEAN_SHIFT_H

#include "../Core/Base.h"
#include "../Core/Validation.h"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace Skigen {

/// @defgroup Algo_MeanShift MeanShift
/// @ingroup Cluster
/// @brief Mode-seeking clustering by iterative mean shifts.
/// @{

/// @brief Mean shift clustering with a flat kernel.
///
/// MeanShift treats each training row as an initial seed, repeatedly moves it
/// to the mean of samples inside the bandwidth ball, then merges converged
/// seeds into cluster centers. Labels are assigned by nearest center, and
/// `cluster_all=false` labels samples outside every bandwidth ball as `-1`.
///
/// Mirrors the dense, brute-force subset of
/// [sklearn.cluster.MeanShift](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html).
///
/// ### Examples
///
/// @snippet mean_shift.cpp example_mean_shift
template <typename Scalar = double>
class MeanShift : public Estimator<MeanShift<Scalar>, Scalar> {
public:
    using Base = Estimator<MeanShift<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::IndexType;
    using IndexVector = Eigen::VectorXi;

    /// @brief Construct a MeanShift estimator.
    ///
    /// @param bandwidth Radius of the flat kernel (`Scalar`, default `1.0`).
    /// @param max_iter Maximum mean-shift iterations per seed (`int`, default `300`).
    /// @param cluster_all Assign every sample to the nearest center when true (`bool`, default `true`).
    explicit MeanShift(Scalar bandwidth = Scalar{1}, int max_iter = 300, bool cluster_all = true)
        : bandwidth_(bandwidth), max_iter_(max_iter), cluster_all_(cluster_all) {}

    /// @brief Bandwidth radius.
    [[nodiscard]] Scalar bandwidth() const noexcept { return bandwidth_; }

    /// @brief Maximum iterations per seed.
    [[nodiscard]] int max_iter() const noexcept { return max_iter_; }

    /// @brief Whether every sample is assigned to its nearest center.
    [[nodiscard]] bool cluster_all() const noexcept { return cluster_all_; }

    /// @brief Cluster centers, shape `(n_clusters, n_features)`.
    [[nodiscard]] const MatrixType& cluster_centers() const {
        this->check_is_fitted();
        return cluster_centers_;
    }

    /// @brief Label for each training sample.
    [[nodiscard]] const IndexVector& labels() const {
        this->check_is_fitted();
        return labels_;
    }

    /// @brief Maximum number of iterations used by any converged seed.
    [[nodiscard]] int n_iter() const {
        this->check_is_fitted();
        return n_iter_;
    }

    SKIGEN_PARAMS(
        (bandwidth, bandwidth_, double),
        (max_iter, max_iter_, int),
        (cluster_all, cluster_all_, bool))

    /// @brief Fit mean-shift clustering from a dense design matrix.
    MeanShift& fit(const Eigen::Ref<const MatrixType>& X) {
        internal::check_non_empty(X);
        validate_parameters();

        std::vector<ShiftedSeed> shifted_seeds;
        shifted_seeds.reserve(static_cast<std::size_t>(X.rows()));
        n_iter_ = 0;

        for (Eigen::Index sample_index = 0; sample_index < X.rows(); ++sample_index) {
            MatrixType center = X.row(sample_index);
            int iterations = 0;
            for (int iteration = 0; iteration < max_iter_; ++iteration) {
                const MatrixType next_center = mean_inside_bandwidth(X, center);
                const Scalar shift = (next_center.row(0) - center.row(0)).norm();
                center = next_center;
                iterations = iteration + 1;
                if (shift <= convergence_tolerance()) break;
            }
            const int support = count_inside_bandwidth(X, center);
            if (support > 0) {
                shifted_seeds.push_back(ShiftedSeed{center, support});
                n_iter_ = std::max(n_iter_, iterations);
            }
        }

        std::sort(shifted_seeds.begin(), shifted_seeds.end(), [](const ShiftedSeed& left, const ShiftedSeed& right) {
            if (left.support != right.support) return left.support > right.support;
            return lexicographically_less(left.center, right.center);
        });

        std::vector<MatrixType> centers;
        centers.reserve(shifted_seeds.size());
        for (const auto& seed : shifted_seeds) {
            bool duplicate = false;
            for (const auto& center : centers) {
                if ((seed.center.row(0) - center.row(0)).squaredNorm() <= bandwidth_ * bandwidth_) {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) centers.push_back(seed.center);
        }

        cluster_centers_.resize(static_cast<Eigen::Index>(centers.size()), X.cols());
        for (Eigen::Index center_index = 0; center_index < cluster_centers_.rows(); ++center_index) {
            cluster_centers_.row(center_index) = centers[static_cast<std::size_t>(center_index)].row(0);
        }

        labels_ = assign_labels(X);
        this->n_features_in_ = X.cols();
        this->fitted_ = true;
        return *this;
    }

    /// @brief Fit and return labels for the training samples.
    [[nodiscard]] IndexVector fit_predict(const Eigen::Ref<const MatrixType>& X) {
        fit(X);
        return labels_;
    }

    /// @brief Predict the nearest fitted mean-shift cluster for each sample.
    [[nodiscard]] IndexVector predict(const Eigen::Ref<const MatrixType>& X) const {
        this->check_is_fitted();
        this->validate_feature_count(X);
        return assign_labels(X);
    }

private:
    struct ShiftedSeed {
        MatrixType center;
        int support;
    };

    Scalar bandwidth_;
    int max_iter_;
    bool cluster_all_;
    MatrixType cluster_centers_;
    IndexVector labels_;
    int n_iter_ = 0;

    void validate_parameters() const {
        if (bandwidth_ <= Scalar{0}) {
            throw std::invalid_argument("MeanShift: bandwidth must be positive.");
        }
        if (max_iter_ <= 0) {
            throw std::invalid_argument("MeanShift: max_iter must be positive.");
        }
    }

    [[nodiscard]] Scalar convergence_tolerance() const {
        return std::max(bandwidth_ * Scalar{1e-3}, std::numeric_limits<Scalar>::epsilon());
    }

    [[nodiscard]] MatrixType mean_inside_bandwidth(const Eigen::Ref<const MatrixType>& X,
                                                   const MatrixType& center) const {
        MatrixType mean = MatrixType::Zero(1, X.cols());
        Eigen::Index count = 0;
        const Scalar radius_squared = bandwidth_ * bandwidth_;
        for (Eigen::Index sample_index = 0; sample_index < X.rows(); ++sample_index) {
            if ((X.row(sample_index) - center.row(0)).squaredNorm() <= radius_squared) {
                mean.row(0) += X.row(sample_index);
                ++count;
            }
        }
        if (count == 0) return center;
        mean /= static_cast<Scalar>(count);
        return mean;
    }

    [[nodiscard]] int count_inside_bandwidth(const Eigen::Ref<const MatrixType>& X,
                                             const MatrixType& center) const {
        int count = 0;
        const Scalar radius_squared = bandwidth_ * bandwidth_;
        for (Eigen::Index sample_index = 0; sample_index < X.rows(); ++sample_index) {
            if ((X.row(sample_index) - center.row(0)).squaredNorm() <= radius_squared) ++count;
        }
        return count;
    }

    [[nodiscard]] IndexVector assign_labels(const Eigen::Ref<const MatrixType>& X) const {
        IndexVector labels(X.rows());
        const Scalar radius_squared = bandwidth_ * bandwidth_;
        for (Eigen::Index sample_index = 0; sample_index < X.rows(); ++sample_index) {
            Scalar best_distance = std::numeric_limits<Scalar>::max();
            int best_cluster = -1;
            for (Eigen::Index center_index = 0; center_index < cluster_centers_.rows(); ++center_index) {
                const Scalar distance = (X.row(sample_index) - cluster_centers_.row(center_index)).squaredNorm();
                if (distance < best_distance) {
                    best_distance = distance;
                    best_cluster = static_cast<int>(center_index);
                }
            }
            labels(sample_index) = (cluster_all_ || best_distance <= radius_squared) ? best_cluster : -1;
        }
        return labels;
    }

    static bool lexicographically_less(const MatrixType& left, const MatrixType& right) {
        for (Eigen::Index col = 0; col < left.cols(); ++col) {
            if (left(0, col) < right(0, col)) return true;
            if (right(0, col) < left(0, col)) return false;
        }
        return false;
    }
};

/// @}

} // namespace Skigen

#endif // SKIGEN_CLUSTER_MEAN_SHIFT_H