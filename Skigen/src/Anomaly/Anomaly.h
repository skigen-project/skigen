// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

#ifndef SKIGEN_ANOMALY_ANOMALY_H
#define SKIGEN_ANOMALY_ANOMALY_H

#include "../Core/Base.h"
#include "../Core/Validation.h"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace Skigen {

/// @defgroup Algo_Anomaly Anomaly Detection
/// @ingroup Anomaly
/// @brief Dense outlier and novelty detection estimators.
/// @{

namespace anomaly::detail {

template <typename Scalar>
Scalar average_path_length(Eigen::Index n_samples) {
    if (n_samples <= 1) {
        return Scalar{0};
    }
    if (n_samples == 2) {
        return Scalar{1};
    }
    const Scalar n = static_cast<Scalar>(n_samples);
    constexpr Scalar euler_gamma = Scalar{0.5772156649015328606L};
    return Scalar{2} * (std::log(n - Scalar{1}) + euler_gamma) -
           Scalar{2} * (n - Scalar{1}) / n;
}

template <typename Scalar>
Scalar score_quantile(Eigen::Matrix<Scalar, Eigen::Dynamic, 1> scores,
                      Scalar contamination) {
    std::sort(scores.data(), scores.data() + scores.size());
    const Eigen::Index index = std::clamp(
        static_cast<Eigen::Index>(std::floor(contamination * static_cast<Scalar>(scores.size()))),
        Eigen::Index{0},
        scores.size() - 1);
    return scores(index);
}

template <typename Scalar>
void validate_contamination(Scalar contamination, const char* estimator_name) {
    if (!(contamination > Scalar{0} && contamination <= Scalar{0.5})) {
        throw std::invalid_argument(
            std::string(estimator_name) + ": contamination must be in (0, 0.5].");
    }
}

}  // namespace anomaly::detail

/// @brief Robust Gaussian-envelope outlier detector for dense numeric input.
///
/// Fits an empirical covariance model, scores samples by negative robust
/// Mahalanobis distance, and thresholds scores by the requested contamination.
///
/// Mirrors the dense core of `sklearn.covariance.EllipticEnvelope`:
/// https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EllipticEnvelope.html.
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `contamination` | `Scalar` | `0.1` | Expected outlier fraction in `(0, 0.5]`. |
/// | `assume_centered` | `bool` | `false` | If `true`, data is assumed centered. |
/// | `regularization` | `Scalar` | `1e-9` | Diagonal covariance regularization. |
///
/// ### Examples
///
/// @snippet anomaly.cpp example_elliptic_envelope
template <typename Scalar = double>
class EllipticEnvelope : public Estimator<EllipticEnvelope<Scalar>, Scalar> {
public:
    using Base = Estimator<EllipticEnvelope<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;

    explicit EllipticEnvelope(Scalar contamination = Scalar{0.1},
                              bool assume_centered = false,
                              Scalar regularization = Scalar{1e-9})
        : contamination_(contamination),
          assume_centered_(assume_centered),
          regularization_(regularization) {}

    [[nodiscard]] Scalar contamination() const noexcept { return contamination_; }
    [[nodiscard]] bool assume_centered() const noexcept { return assume_centered_; }
    [[nodiscard]] Scalar regularization() const noexcept { return regularization_; }

    [[nodiscard]] const RowVectorType& location() const {
        this->check_is_fitted();
        return location_;
    }

    [[nodiscard]] const MatrixType& covariance() const {
        this->check_is_fitted();
        return covariance_;
    }

    [[nodiscard]] const MatrixType& precision() const {
        this->check_is_fitted();
        return precision_;
    }

    [[nodiscard]] Scalar offset() const {
        this->check_is_fitted();
        return offset_;
    }

    [[nodiscard]] const VectorType& dist() const {
        this->check_is_fitted();
        return dist_;
    }

    SKIGEN_PARAMS(
        (contamination, contamination_, double),
        (assume_centered, assume_centered_, bool),
        (regularization, regularization_, double))

    EllipticEnvelope& fit(const Eigen::Ref<const MatrixType>& input) {
        internal::check_non_empty(input);
        if (!input.allFinite()) {
            throw std::invalid_argument("EllipticEnvelope: X must contain only finite values.");
        }
        anomaly::detail::validate_contamination(contamination_, "EllipticEnvelope");
        if (regularization_ < Scalar{0}) {
            throw std::invalid_argument("EllipticEnvelope: regularization must be non-negative.");
        }

        this->n_features_in_ = input.cols();
        if (assume_centered_) {
            location_ = RowVectorType::Zero(input.cols());
        } else {
            location_ = input.colwise().mean();
        }
        const MatrixType centered = input.rowwise() - location_;
        covariance_ = centered.transpose() * centered / static_cast<Scalar>(input.rows());
        covariance_.diagonal().array() += regularization_;

        Eigen::SelfAdjointEigenSolver<MatrixType> solver(covariance_);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("EllipticEnvelope: covariance eigensolve failed.");
        }
        const auto evals = solver.eigenvalues().array().max(Scalar{1e-30}).eval();
        precision_ = solver.eigenvectors() * evals.inverse().matrix().asDiagonal() *
                     solver.eigenvectors().transpose();

        this->fitted_ = true;
        dist_ = mahalanobis(input);
        offset_ = anomaly::detail::score_quantile(score_samples(input), contamination_);
        return *this;
    }

    [[nodiscard]] VectorType score_samples(const Eigen::Ref<const MatrixType>& input) const {
        this->check_is_fitted();
        this->validate_feature_count(input);
        return -mahalanobis(input);
    }

    [[nodiscard]] VectorType decision_function(const Eigen::Ref<const MatrixType>& input) const {
        return (score_samples(input).array() - offset_).matrix();
    }

    [[nodiscard]] Eigen::VectorXi predict(const Eigen::Ref<const MatrixType>& input) const {
        const VectorType decisions = decision_function(input);
        Eigen::VectorXi labels(input.rows());
        for (IndexType row = 0; row < input.rows(); ++row) {
            labels(row) = decisions(row) >= Scalar{0} ? 1 : -1;
        }
        return labels;
    }

private:
    [[nodiscard]] VectorType mahalanobis(const Eigen::Ref<const MatrixType>& input) const {
        const MatrixType centered = input.rowwise() - location_;
        VectorType distances(input.rows());
        for (IndexType row = 0; row < input.rows(); ++row) {
            distances(row) = centered.row(row) * precision_ * centered.row(row).transpose();
        }
        return distances;
    }

    Scalar contamination_;
    bool assume_centered_;
    Scalar regularization_;
    RowVectorType location_;
    MatrixType covariance_;
    MatrixType precision_;
    VectorType dist_;
    Scalar offset_{0};
};

/// @brief Random isolation-tree ensemble for dense anomaly detection.
///
/// Builds random split trees and scores samples by average path length. Larger
/// `score_samples` values indicate more normal observations; `predict` returns
/// `1` for inliers and `-1` for outliers.
///
/// Mirrors the dense core of `sklearn.ensemble.IsolationForest`:
/// https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html.
///
/// ### Examples
///
/// @snippet anomaly.cpp example_isolation_forest
template <typename Scalar = double>
class IsolationForest : public Estimator<IsolationForest<Scalar>, Scalar> {
public:
    using Base = Estimator<IsolationForest<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;

    explicit IsolationForest(int n_estimators = 100,
                             int max_samples = 256,
                             Scalar contamination = Scalar{0.1},
                             int max_depth = -1,
                             int random_state = -1)
        : n_estimators_(n_estimators),
          max_samples_(max_samples),
          contamination_(contamination),
          max_depth_(max_depth),
          random_state_(random_state) {}

    [[nodiscard]] int n_estimators() const noexcept { return n_estimators_; }
    [[nodiscard]] int max_samples() const noexcept { return max_samples_; }
    [[nodiscard]] Scalar contamination() const noexcept { return contamination_; }
    [[nodiscard]] int max_depth() const noexcept { return max_depth_; }
    [[nodiscard]] int random_state() const noexcept { return random_state_; }
    [[nodiscard]] int max_samples_effective() const noexcept { return max_samples_eff_; }

    [[nodiscard]] Scalar offset() const {
        this->check_is_fitted();
        return offset_;
    }

    SKIGEN_PARAMS(
        (n_estimators, n_estimators_, int),
        (max_samples, max_samples_, int),
        (contamination, contamination_, double),
        (max_depth, max_depth_, int),
        (random_state, random_state_, int))

    IsolationForest& fit(const Eigen::Ref<const MatrixType>& input) {
        internal::check_non_empty(input);
        if (!input.allFinite()) {
            throw std::invalid_argument("IsolationForest: X must contain only finite values.");
        }
        validate_parameters(input.rows());
        this->n_features_in_ = input.cols();
        max_samples_eff_ = std::min(max_samples_, static_cast<int>(input.rows()));
        depth_limit_ = max_depth_ >= 0
            ? max_depth_
            : static_cast<int>(std::ceil(std::log2(static_cast<double>(max_samples_eff_))));

        std::mt19937_64 rng = make_rng();
        trees_.clear();
        trees_.reserve(static_cast<std::size_t>(n_estimators_));
        std::vector<IndexType> all_indices(static_cast<std::size_t>(input.rows()));
        std::iota(all_indices.begin(), all_indices.end(), IndexType{0});
        for (int estimator = 0; estimator < n_estimators_; ++estimator) {
            std::shuffle(all_indices.begin(), all_indices.end(), rng);
            std::vector<IndexType> sample(
                all_indices.begin(), all_indices.begin() + max_samples_eff_);
            trees_.push_back(Tree{build_tree(input, sample, 0, rng)});
        }

        this->fitted_ = true;
        offset_ = anomaly::detail::score_quantile(score_samples(input), contamination_);
        return *this;
    }

    [[nodiscard]] VectorType score_samples(const Eigen::Ref<const MatrixType>& input) const {
        this->check_is_fitted();
        this->validate_feature_count(input);
        const Scalar normalizer = anomaly::detail::average_path_length<Scalar>(max_samples_eff_);
        VectorType scores(input.rows());
        for (IndexType row = 0; row < input.rows(); ++row) {
            Scalar path_sum = Scalar{0};
            for (const Tree& tree : trees_) {
                path_sum += path_length(tree.root.get(), input.row(row), 0);
            }
            const Scalar average_path = path_sum / static_cast<Scalar>(trees_.size());
            const Scalar anomaly_score = std::pow(Scalar{2}, -average_path / normalizer);
            scores(row) = -anomaly_score;
        }
        return scores;
    }

    [[nodiscard]] VectorType decision_function(const Eigen::Ref<const MatrixType>& input) const {
        return (score_samples(input).array() - offset_).matrix();
    }

    [[nodiscard]] Eigen::VectorXi predict(const Eigen::Ref<const MatrixType>& input) const {
        const VectorType decisions = decision_function(input);
        Eigen::VectorXi labels(input.rows());
        for (IndexType row = 0; row < input.rows(); ++row) {
            labels(row) = decisions(row) >= Scalar{0} ? 1 : -1;
        }
        return labels;
    }

private:
    struct Node {
        bool is_leaf = true;
        IndexType feature = -1;
        Scalar threshold{0};
        Eigen::Index n_samples = 0;
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;
    };

    struct Tree {
        std::unique_ptr<Node> root;
    };

    void validate_parameters(IndexType n_samples) const {
        if (n_estimators_ <= 0) {
            throw std::invalid_argument("IsolationForest: n_estimators must be positive.");
        }
        if (max_samples_ <= 0) {
            throw std::invalid_argument("IsolationForest: max_samples must be positive.");
        }
        if (max_samples_ > n_samples) {
            throw std::invalid_argument("IsolationForest: max_samples cannot exceed n_samples.");
        }
        anomaly::detail::validate_contamination(contamination_, "IsolationForest");
        if (max_depth_ == 0 || max_depth_ < -1) {
            throw std::invalid_argument("IsolationForest: max_depth must be -1 or positive.");
        }
    }

    [[nodiscard]] std::mt19937_64 make_rng() const {
        if (random_state_ >= 0) {
            return std::mt19937_64(static_cast<std::uint64_t>(random_state_));
        }
        std::random_device device;
        return std::mt19937_64(device());
    }

    [[nodiscard]] std::unique_ptr<Node> build_tree(const MatrixType& input,
                                                   const std::vector<IndexType>& indices,
                                                   int depth,
                                                   std::mt19937_64& rng) const {
        auto node = std::make_unique<Node>();
        node->n_samples = static_cast<Eigen::Index>(indices.size());
        if (depth >= depth_limit_ || indices.size() <= 1) {
            return node;
        }

        std::vector<IndexType> varying_features;
        varying_features.reserve(static_cast<std::size_t>(input.cols()));
        std::vector<Scalar> min_values(static_cast<std::size_t>(input.cols()));
        std::vector<Scalar> max_values(static_cast<std::size_t>(input.cols()));
        for (IndexType feature = 0; feature < input.cols(); ++feature) {
            Scalar min_value = input(indices.front(), feature);
            Scalar max_value = min_value;
            for (IndexType row : indices) {
                min_value = std::min(min_value, input(row, feature));
                max_value = std::max(max_value, input(row, feature));
            }
            min_values[static_cast<std::size_t>(feature)] = min_value;
            max_values[static_cast<std::size_t>(feature)] = max_value;
            if (max_value > min_value) {
                varying_features.push_back(feature);
            }
        }
        if (varying_features.empty()) {
            return node;
        }

        std::uniform_int_distribution<std::size_t> feature_dist(0, varying_features.size() - 1);
        const IndexType feature = varying_features[feature_dist(rng)];
        const Scalar min_value = min_values[static_cast<std::size_t>(feature)];
        const Scalar max_value = max_values[static_cast<std::size_t>(feature)];
        std::uniform_real_distribution<double> threshold_dist(
            static_cast<double>(min_value), static_cast<double>(max_value));
        const Scalar threshold = static_cast<Scalar>(threshold_dist(rng));

        std::vector<IndexType> left_indices;
        std::vector<IndexType> right_indices;
        left_indices.reserve(indices.size());
        right_indices.reserve(indices.size());
        for (IndexType row : indices) {
            if (input(row, feature) < threshold) {
                left_indices.push_back(row);
            } else {
                right_indices.push_back(row);
            }
        }
        if (left_indices.empty() || right_indices.empty()) {
            return node;
        }

        node->is_leaf = false;
        node->feature = feature;
        node->threshold = threshold;
        node->left = build_tree(input, left_indices, depth + 1, rng);
        node->right = build_tree(input, right_indices, depth + 1, rng);
        return node;
    }

    [[nodiscard]] Scalar path_length(const Node* node,
                                     const Eigen::Ref<const RowVectorType>& row,
                                     int depth) const {
        if (node->is_leaf) {
            return static_cast<Scalar>(depth) +
                   anomaly::detail::average_path_length<Scalar>(node->n_samples);
        }
        if (row(node->feature) < node->threshold) {
            return path_length(node->left.get(), row, depth + 1);
        }
        return path_length(node->right.get(), row, depth + 1);
    }

    int n_estimators_;
    int max_samples_;
    Scalar contamination_;
    int max_depth_;
    int random_state_;
    int max_samples_eff_ = 0;
    int depth_limit_ = 0;
    Scalar offset_{0};
    std::vector<Tree> trees_;
};

/// @}

}  // namespace Skigen

#endif  // SKIGEN_ANOMALY_ANOMALY_H
