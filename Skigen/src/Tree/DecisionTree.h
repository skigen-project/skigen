// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_TREE_DECISION_TREE_H
#define SKIGEN_TREE_DECISION_TREE_H

#include "../Core/Validation.h"

#include <Eigen/Core>
#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace Skigen {

namespace internal {

template <typename Scalar>
struct TreeNode {
    bool is_leaf = false;
    Eigen::Index feature_index = -1;
    Scalar threshold{0};
    int prediction_class = -1;   // for classifier
    Scalar prediction_value{0};  // for regressor
    std::unique_ptr<TreeNode> left;
    std::unique_ptr<TreeNode> right;
};

} // namespace internal

/// DecisionTreeClassifier — CART decision tree for classification.
/// Uses Gini impurity. Mirrors sklearn.tree.DecisionTreeClassifier.
template <typename Scalar = double>
class DecisionTreeClassifier {
public:
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using IndexVector = Eigen::VectorXi;

    explicit DecisionTreeClassifier(int max_depth = -1, int min_samples_split = 2)
        : max_depth_(max_depth), min_samples_split_(min_samples_split) {}

    [[nodiscard]] bool is_fitted() const noexcept { return fitted_; }

    DecisionTreeClassifier& fit(const Eigen::Ref<const MatrixType>& X,
                                const Eigen::Ref<const IndexVector>& y) {
        internal::check_non_empty(X);
        if (X.rows() != y.rows()) {
            throw std::invalid_argument("X and y have inconsistent lengths.");
        }

        n_features_in_ = X.cols();

        // Collect class labels
        std::map<int, int> class_counts;
        for (Eigen::Index i = 0; i < y.size(); ++i) {
            class_counts[y(i)]++;
        }

        std::vector<Eigen::Index> indices(static_cast<std::size_t>(X.rows()));
        std::iota(indices.begin(), indices.end(), Eigen::Index{0});

        root_ = build_tree(X, y, indices, 0);
        fitted_ = true;
        return *this;
    }

    [[nodiscard]] IndexVector predict(
        const Eigen::Ref<const MatrixType>& X) const {
        if (!fitted_) throw std::runtime_error(
            "DecisionTreeClassifier has not been fitted yet.");

        IndexVector predictions(X.rows());
        for (Eigen::Index i = 0; i < X.rows(); ++i) {
            predictions(i) = predict_one(root_.get(), X.row(i));
        }
        return predictions;
    }

    [[nodiscard]] Scalar score(const Eigen::Ref<const MatrixType>& X,
                               const Eigen::Ref<const IndexVector>& y) const {
        IndexVector preds = predict(X);
        int correct = 0;
        for (Eigen::Index i = 0; i < y.size(); ++i) {
            if (preds(i) == y(i)) ++correct;
        }
        return static_cast<Scalar>(correct) / static_cast<Scalar>(y.size());
    }

private:
    int max_depth_;
    int min_samples_split_;
    bool fitted_ = false;
    Eigen::Index n_features_in_ = 0;
    std::unique_ptr<internal::TreeNode<Scalar>> root_;

    using Node = internal::TreeNode<Scalar>;

    std::unique_ptr<Node> build_tree(
        const Eigen::Ref<const MatrixType>& X,
        const Eigen::Ref<const IndexVector>& y,
        const std::vector<Eigen::Index>& indices,
        int depth) const {

        auto node = std::make_unique<Node>();
        const auto n = static_cast<int>(indices.size());

        // Majority class
        std::map<int, int> counts;
        for (auto idx : indices) counts[y(idx)]++;
        int majority = counts.begin()->first;
        int majority_count = counts.begin()->second;
        for (const auto& [cls, cnt] : counts) {
            if (cnt > majority_count) {
                majority = cls;
                majority_count = cnt;
            }
        }

        // Leaf conditions
        if (counts.size() == 1 || n < min_samples_split_ ||
            (max_depth_ >= 0 && depth >= max_depth_)) {
            node->is_leaf = true;
            node->prediction_class = majority;
            return node;
        }

        // Find best split
        Scalar best_gini = std::numeric_limits<Scalar>::max();
        Eigen::Index best_feature = 0;
        Scalar best_threshold{0};
        std::vector<Eigen::Index> best_left, best_right;

        for (Eigen::Index f = 0; f < X.cols(); ++f) {
            // Collect unique thresholds
            std::vector<Scalar> values;
            values.reserve(indices.size());
            for (auto idx : indices) values.push_back(X(idx, f));
            std::sort(values.begin(), values.end());
            values.erase(std::unique(values.begin(), values.end()), values.end());

            for (std::size_t t = 0; t + 1 < values.size(); ++t) {
                Scalar thresh = (values[t] + values[t + 1]) / Scalar{2};

                std::vector<Eigen::Index> left_idx, right_idx;
                for (auto idx : indices) {
                    if (X(idx, f) <= thresh) left_idx.push_back(idx);
                    else right_idx.push_back(idx);
                }

                if (left_idx.empty() || right_idx.empty()) continue;

                Scalar gini = weighted_gini(y, left_idx, right_idx);
                if (gini < best_gini) {
                    best_gini = gini;
                    best_feature = f;
                    best_threshold = thresh;
                    best_left = std::move(left_idx);
                    best_right = std::move(right_idx);
                }
            }
        }

        if (best_left.empty()) {
            node->is_leaf = true;
            node->prediction_class = majority;
            return node;
        }

        node->feature_index = best_feature;
        node->threshold = best_threshold;
        node->left = build_tree(X, y, best_left, depth + 1);
        node->right = build_tree(X, y, best_right, depth + 1);

        return node;
    }

    static Scalar gini_impurity(const Eigen::Ref<const IndexVector>& y,
                                const std::vector<Eigen::Index>& indices) {
        std::map<int, int> counts;
        for (auto idx : indices) counts[y(idx)]++;
        Scalar n = static_cast<Scalar>(indices.size());
        Scalar impurity{1};
        for (const auto& [cls, cnt] : counts) {
            Scalar p = static_cast<Scalar>(cnt) / n;
            impurity -= p * p;
        }
        return impurity;
    }

    static Scalar weighted_gini(const Eigen::Ref<const IndexVector>& y,
                                const std::vector<Eigen::Index>& left,
                                const std::vector<Eigen::Index>& right) {
        Scalar n = static_cast<Scalar>(left.size() + right.size());
        return (static_cast<Scalar>(left.size()) * gini_impurity(y, left) +
                static_cast<Scalar>(right.size()) * gini_impurity(y, right)) / n;
    }

    template <typename RowExpr>
    static int predict_one(const Node* node, const RowExpr& x) {
        if (node->is_leaf) return node->prediction_class;
        if (x(node->feature_index) <= node->threshold) {
            return predict_one(node->left.get(), x);
        }
        return predict_one(node->right.get(), x);
    }
};

/// DecisionTreeRegressor — CART decision tree for regression.
/// Uses MSE reduction. Mirrors sklearn.tree.DecisionTreeRegressor.
template <typename Scalar = double>
class DecisionTreeRegressor {
public:
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    explicit DecisionTreeRegressor(int max_depth = -1, int min_samples_split = 2)
        : max_depth_(max_depth), min_samples_split_(min_samples_split) {}

    [[nodiscard]] bool is_fitted() const noexcept { return fitted_; }

    DecisionTreeRegressor& fit(const Eigen::Ref<const MatrixType>& X,
                               const Eigen::Ref<const VectorType>& y) {
        internal::check_non_empty(X);
        if (X.rows() != y.rows()) {
            throw std::invalid_argument("X and y have inconsistent lengths.");
        }

        n_features_in_ = X.cols();

        std::vector<Eigen::Index> indices(static_cast<std::size_t>(X.rows()));
        std::iota(indices.begin(), indices.end(), Eigen::Index{0});

        root_ = build_tree(X, y, indices, 0);
        fitted_ = true;
        return *this;
    }

    [[nodiscard]] VectorType predict(
        const Eigen::Ref<const MatrixType>& X) const {
        if (!fitted_) throw std::runtime_error(
            "DecisionTreeRegressor has not been fitted yet.");

        VectorType predictions(X.rows());
        for (Eigen::Index i = 0; i < X.rows(); ++i) {
            predictions(i) = predict_one(root_.get(), X.row(i));
        }
        return predictions;
    }

    [[nodiscard]] Scalar score(const Eigen::Ref<const MatrixType>& X,
                               const Eigen::Ref<const VectorType>& y) const {
        VectorType y_pred = predict(X);
        Scalar ss_res = (y - y_pred).squaredNorm();
        Scalar ss_tot = (y.array() - y.mean()).matrix().squaredNorm();
        if (ss_tot == Scalar{0}) return Scalar{0};
        return Scalar{1} - ss_res / ss_tot;
    }

private:
    int max_depth_;
    int min_samples_split_;
    bool fitted_ = false;
    Eigen::Index n_features_in_ = 0;
    std::unique_ptr<internal::TreeNode<Scalar>> root_;

    using Node = internal::TreeNode<Scalar>;

    std::unique_ptr<Node> build_tree(
        const Eigen::Ref<const MatrixType>& X,
        const Eigen::Ref<const VectorType>& y,
        const std::vector<Eigen::Index>& indices,
        int depth) const {

        auto node = std::make_unique<Node>();
        const auto n = static_cast<int>(indices.size());

        // Mean value
        Scalar mean_val{0};
        for (auto idx : indices) mean_val += y(idx);
        mean_val /= static_cast<Scalar>(n);

        // Leaf conditions
        if (n < min_samples_split_ ||
            (max_depth_ >= 0 && depth >= max_depth_)) {
            node->is_leaf = true;
            node->prediction_value = mean_val;
            return node;
        }

        // Find best split by MSE reduction
        Scalar best_mse = std::numeric_limits<Scalar>::max();
        Eigen::Index best_feature = 0;
        Scalar best_threshold{0};
        std::vector<Eigen::Index> best_left, best_right;

        for (Eigen::Index f = 0; f < X.cols(); ++f) {
            std::vector<Scalar> values;
            values.reserve(indices.size());
            for (auto idx : indices) values.push_back(X(idx, f));
            std::sort(values.begin(), values.end());
            values.erase(std::unique(values.begin(), values.end()), values.end());

            for (std::size_t t = 0; t + 1 < values.size(); ++t) {
                Scalar thresh = (values[t] + values[t + 1]) / Scalar{2};

                std::vector<Eigen::Index> left_idx, right_idx;
                for (auto idx : indices) {
                    if (X(idx, f) <= thresh) left_idx.push_back(idx);
                    else right_idx.push_back(idx);
                }

                if (left_idx.empty() || right_idx.empty()) continue;

                Scalar mse = weighted_mse(y, left_idx, right_idx);
                if (mse < best_mse) {
                    best_mse = mse;
                    best_feature = f;
                    best_threshold = thresh;
                    best_left = std::move(left_idx);
                    best_right = std::move(right_idx);
                }
            }
        }

        if (best_left.empty()) {
            node->is_leaf = true;
            node->prediction_value = mean_val;
            return node;
        }

        node->feature_index = best_feature;
        node->threshold = best_threshold;
        node->left = build_tree(X, y, best_left, depth + 1);
        node->right = build_tree(X, y, best_right, depth + 1);

        return node;
    }

    static Scalar mse(const Eigen::Ref<const VectorType>& y,
                      const std::vector<Eigen::Index>& indices) {
        Scalar mean{0};
        for (auto idx : indices) mean += y(idx);
        mean /= static_cast<Scalar>(indices.size());

        Scalar s{0};
        for (auto idx : indices) {
            Scalar d = y(idx) - mean;
            s += d * d;
        }
        return s / static_cast<Scalar>(indices.size());
    }

    static Scalar weighted_mse(const Eigen::Ref<const VectorType>& y,
                               const std::vector<Eigen::Index>& left,
                               const std::vector<Eigen::Index>& right) {
        Scalar n = static_cast<Scalar>(left.size() + right.size());
        return (static_cast<Scalar>(left.size()) * mse(y, left) +
                static_cast<Scalar>(right.size()) * mse(y, right)) / n;
    }

    template <typename RowExpr>
    static Scalar predict_one(const Node* node, const RowExpr& x) {
        if (node->is_leaf) return node->prediction_value;
        if (x(node->feature_index) <= node->threshold) {
            return predict_one(node->left.get(), x);
        }
        return predict_one(node->right.get(), x);
    }
};

} // namespace Skigen

#endif // SKIGEN_TREE_DECISION_TREE_H
