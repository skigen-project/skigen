// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_TREE_DECISION_TREE_H
#define SKIGEN_TREE_DECISION_TREE_H

#include "../Core/Base.h"
#include "../Core/Validation.h"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
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
    // For classifier: per-class sample counts at this leaf (length == n_classes_).
    std::vector<Scalar> class_counts;
    // Number of (training) samples seen at this node.
    Scalar n_node_samples{0};
    // Impurity at this node (Gini or MSE).
    Scalar impurity{0};
    // Best impurity decrease produced by the split (0 for leaves).
    Scalar impurity_decrease{0};
    std::unique_ptr<TreeNode> left;
    std::unique_ptr<TreeNode> right;
};

// Helper: compute the effective number of features to consider at each split.
// `mode_value` interpretation:
//   0 = All           — use n_features
//   1 = Sqrt          — floor(sqrt(n_features))
//   2 = Log2          — floor(log2(n_features))
//   3 = FractionOrCount — uses `value` (>=1 → cast to int; in (0,1] → fraction)
inline int compute_max_features_eff(int mode, double value, int n_features) {
    int m = n_features;
    switch (mode) {
        case 0: m = n_features; break;
        case 1: m = std::max(1, static_cast<int>(std::floor(std::sqrt(static_cast<double>(n_features))))); break;
        case 2: m = std::max(1, static_cast<int>(std::floor(std::log2(static_cast<double>(n_features))))); break;
        case 3: {
            if (value >= 1.0) m = static_cast<int>(value);
            else if (value > 0.0) m = std::max(1, static_cast<int>(std::ceil(value * static_cast<double>(n_features))));
            else m = n_features;
            break;
        }
        default: m = n_features;
    }
    if (m < 1) m = 1;
    if (m > n_features) m = n_features;
    return m;
}

} // namespace internal

/// @defgroup Algo_DecisionTree Decision Trees
/// @ingroup Tree
/// @brief CART decision trees for classification and regression.
/// @{

/// @brief A decision tree classifier.
///
/// A non-parametric supervised learning method used for classification.
/// The model predicts the value of a target variable by learning
/// simple decision rules inferred from the data features.
/// Uses Gini impurity as the splitting criterion.
///
/// Mirrors
/// [sklearn.tree.DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html).
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `max_depth` | `int` | `-1` | Maximum depth of the tree. `-1` means no limit. |
/// | `min_samples_split` | `int` | `2` | Minimum number of samples required to split a node. |
/// | `max_features_mode` | `int` | `0` | Feature subspace mode: 0=All, 1=Sqrt, 2=Log2, 3=FractionOrCount. |
/// | `max_features_value` | `double` | `0` | Used when `max_features_mode == 3`. |
/// | `random_state` | `optional<uint64_t>` | `nullopt` | RNG seed for feature subsampling. |
///
/// ### Notes
///
/// Uses Gini impurity for classification. The tree is built recursively
/// with greedy best-first splitting.
///
/// @note **scikit-learn parity gaps:** The following sklearn constructor
///   parameters are not yet supported: `criterion` (only Gini),
///   `splitter`, `min_samples_leaf`, `min_weight_fraction_leaf`,
///   `max_leaf_nodes`, `min_impurity_decrease`,
///   `class_weight`, `ccp_alpha`, `monotonic_cst`.
///
/// ### Examples
///
/// @snippet decision_tree.cpp example_decision_tree_classifier
template <typename Scalar = double>
class DecisionTreeClassifier
    : public Classifier<DecisionTreeClassifier<Scalar>, Scalar> {
public:
    using Base = Classifier<DecisionTreeClassifier<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;
    using typename Base::ScalarType;
    using typename Base::LabelType;
    using IndexVector = Eigen::VectorXi;

    /// @brief Construct a DecisionTreeClassifier.
    explicit DecisionTreeClassifier(int max_depth = -1,
                                    int min_samples_split = 2,
                                    int max_features_mode = 0,
                                    double max_features_value = 0.0,
                                    std::optional<uint64_t> random_state = std::nullopt)
        : max_depth_(max_depth),
          min_samples_split_(min_samples_split),
          max_features_mode_(max_features_mode),
          max_features_value_(max_features_value),
          random_state_(random_state) {}

    /// @brief Build a decision tree classifier from the training set.
    DecisionTreeClassifier& fit_impl(const Eigen::Ref<const MatrixType>& X,
                                const Eigen::Ref<const IndexVector>& y) {
        return fit_with_indices(X, y, std::vector<Eigen::Index>{});
    }

    /// @brief Fit using a specific row index subset (used by RandomForest bootstraps).
    /// @param sample_indices If empty, uses all rows of X. Otherwise, builds the
    ///   tree from the rows specified (with possible repetitions).
    DecisionTreeClassifier& fit_with_indices(const Eigen::Ref<const MatrixType>& X,
                                             const Eigen::Ref<const IndexVector>& y,
                                             const std::vector<Eigen::Index>& sample_indices) {
        internal::check_non_empty(X);
        internal::check_consistent_length(X, y);

        this->n_features_in_ = X.cols();

        // Discover unique class labels (sorted ascending).
        std::vector<int> uniq;
        uniq.reserve(static_cast<std::size_t>(y.size()));
        for (Eigen::Index i = 0; i < y.size(); ++i) uniq.push_back(y(i));
        std::sort(uniq.begin(), uniq.end());
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
        classes_ = Eigen::VectorXi(static_cast<Eigen::Index>(uniq.size()));
        class_to_idx_.clear();
        for (std::size_t i = 0; i < uniq.size(); ++i) {
            classes_(static_cast<Eigen::Index>(i)) = uniq[i];
            class_to_idx_[uniq[i]] = static_cast<int>(i);
        }
        n_classes_ = static_cast<int>(uniq.size());

        std::vector<Eigen::Index> indices;
        if (sample_indices.empty()) {
            indices.resize(static_cast<std::size_t>(X.rows()));
            std::iota(indices.begin(), indices.end(), Eigen::Index{0});
        } else {
            indices = sample_indices;
        }

        feature_importances_ = RowVectorType::Zero(X.cols());
        total_weight_ = static_cast<Scalar>(indices.size());

        if (random_state_.has_value()) rng_ = std::mt19937_64(*random_state_);
        else {
            std::random_device rd;
            rng_ = std::mt19937_64(static_cast<uint64_t>(rd()));
        }

        root_ = build_tree(X, y, indices, 0);

        // Normalise feature importances so they sum to 1 (sklearn convention).
        Scalar s = feature_importances_.sum();
        if (s > Scalar{0}) feature_importances_ /= s;

        this->fitted_ = true;
        return *this;
    }

    /// @brief Predict class labels for samples in X.
    [[nodiscard]] IndexVector predict_impl(
        const Eigen::Ref<const MatrixType>& X) const {

        IndexVector predictions(X.rows());
        for (Eigen::Index i = 0; i < X.rows(); ++i) {
            const Node* leaf = walk_to_leaf(root_.get(), X.row(i));
            predictions(i) = classes_(leaf->prediction_class);
        }
        return predictions;
    }

    /// @brief Per-class probability estimates from leaf class distributions.
    [[nodiscard]] MatrixType predict_proba(
        const Eigen::Ref<const MatrixType>& X) const {
        this->check_is_fitted();
        this->validate_feature_count(X);
        MatrixType out(X.rows(), n_classes_);
        for (Eigen::Index i = 0; i < X.rows(); ++i) {
            const Node* leaf = walk_to_leaf(root_.get(), X.row(i));
            Scalar total = Scalar{0};
            for (int c = 0; c < n_classes_; ++c) total += leaf->class_counts[static_cast<std::size_t>(c)];
            for (int c = 0; c < n_classes_; ++c) {
                out(i, c) = total > Scalar{0}
                    ? leaf->class_counts[static_cast<std::size_t>(c)] / total
                    : Scalar{0};
            }
        }
        return out;
    }

    [[nodiscard]] const Eigen::VectorXi& classes() const {
        this->check_is_fitted(); return classes_;
    }
    [[nodiscard]] int n_classes() const {
        this->check_is_fitted(); return n_classes_;
    }
    [[nodiscard]] const RowVectorType& feature_importances() const {
        this->check_is_fitted(); return feature_importances_;
    }

private:
    int max_depth_;
    int min_samples_split_;
    int max_features_mode_;
    double max_features_value_;
    std::optional<uint64_t> random_state_;

    Eigen::VectorXi classes_;
    int n_classes_ = 0;
    std::map<int, int> class_to_idx_;
    RowVectorType feature_importances_;
    Scalar total_weight_{0};
    mutable std::mt19937_64 rng_;

    std::unique_ptr<internal::TreeNode<Scalar>> root_;

    using Node = internal::TreeNode<Scalar>;

    static Scalar gini_from_counts(const std::vector<Scalar>& counts, Scalar n) {
        if (n <= Scalar{0}) return Scalar{0};
        Scalar imp{1};
        for (Scalar c : counts) {
            Scalar p = c / n;
            imp -= p * p;
        }
        return imp;
    }

    std::unique_ptr<Node> build_tree(
        const Eigen::Ref<const MatrixType>& X,
        const Eigen::Ref<const IndexVector>& y,
        const std::vector<Eigen::Index>& indices,
        int depth) {

        auto node = std::make_unique<Node>();
        const auto n = static_cast<int>(indices.size());

        // Per-class counts at this node.
        std::vector<Scalar> counts(static_cast<std::size_t>(n_classes_), Scalar{0});
        for (auto idx : indices) {
            int ci = class_to_idx_.at(y(idx));
            counts[static_cast<std::size_t>(ci)] += Scalar{1};
        }
        // Majority class index (within classes_).
        int majority = 0;
        Scalar majority_count = counts.empty() ? Scalar{0} : counts[0];
        for (std::size_t c = 1; c < counts.size(); ++c) {
            if (counts[c] > majority_count) {
                majority = static_cast<int>(c);
                majority_count = counts[c];
            }
        }
        node->n_node_samples = static_cast<Scalar>(n);
        node->class_counts = counts;
        node->impurity = gini_from_counts(counts, static_cast<Scalar>(n));
        node->prediction_class = majority;

        // Pure node, depth limit, or too few samples → leaf.
        bool pure = node->impurity == Scalar{0};
        if (pure || n < min_samples_split_ ||
            (max_depth_ >= 0 && depth >= max_depth_)) {
            node->is_leaf = true;
            return node;
        }

        // Choose feature subspace.
        const Eigen::Index n_features = X.cols();
        const int max_feat = internal::compute_max_features_eff(
            max_features_mode_, max_features_value_, static_cast<int>(n_features));

        std::vector<Eigen::Index> feat_pool(static_cast<std::size_t>(n_features));
        std::iota(feat_pool.begin(), feat_pool.end(), Eigen::Index{0});

        // Find best split (try features one at a time; if no valid split is
        // found in the random subset we fall through to a leaf — sklearn-like).
        Scalar best_gini = std::numeric_limits<Scalar>::max();
        Eigen::Index best_feature = -1;
        Scalar best_threshold{0};
        std::vector<Eigen::Index> best_left, best_right;
        bool found = false;

        // Shuffle the feature pool once, then take the first max_feat entries
        // (Fisher–Yates is unbiased and reproducible from rng_).
        for (int i = 0; i < max_feat && i < static_cast<int>(feat_pool.size()); ++i) {
            std::uniform_int_distribution<int> dist(i, static_cast<int>(feat_pool.size()) - 1);
            int j = dist(rng_);
            std::swap(feat_pool[static_cast<std::size_t>(i)], feat_pool[static_cast<std::size_t>(j)]);
        }

        for (int fi = 0; fi < max_feat; ++fi) {
            Eigen::Index f = feat_pool[static_cast<std::size_t>(fi)];
            std::vector<Scalar> values;
            values.reserve(indices.size());
            for (auto idx : indices) values.push_back(X(idx, f));
            std::sort(values.begin(), values.end());
            values.erase(std::unique(values.begin(), values.end()), values.end());

            for (std::size_t t = 0; t + 1 < values.size(); ++t) {
                Scalar thresh = (values[t] + values[t + 1]) / Scalar{2};

                std::vector<Eigen::Index> left_idx, right_idx;
                std::vector<Scalar> left_counts(static_cast<std::size_t>(n_classes_), Scalar{0});
                std::vector<Scalar> right_counts(static_cast<std::size_t>(n_classes_), Scalar{0});
                for (auto idx : indices) {
                    int ci = class_to_idx_.at(y(idx));
                    if (X(idx, f) <= thresh) {
                        left_idx.push_back(idx);
                        left_counts[static_cast<std::size_t>(ci)] += Scalar{1};
                    } else {
                        right_idx.push_back(idx);
                        right_counts[static_cast<std::size_t>(ci)] += Scalar{1};
                    }
                }
                if (left_idx.empty() || right_idx.empty()) continue;

                Scalar nl = static_cast<Scalar>(left_idx.size());
                Scalar nr = static_cast<Scalar>(right_idx.size());
                Scalar gini = (nl * gini_from_counts(left_counts, nl) +
                               nr * gini_from_counts(right_counts, nr)) /
                              static_cast<Scalar>(n);

                if (gini < best_gini) {
                    best_gini = gini;
                    best_feature = f;
                    best_threshold = thresh;
                    best_left = std::move(left_idx);
                    best_right = std::move(right_idx);
                    found = true;
                }
            }
        }

        if (!found) {
            node->is_leaf = true;
            return node;
        }

        node->feature_index = best_feature;
        node->threshold = best_threshold;
        Scalar dec = node->impurity - best_gini;
        if (dec < Scalar{0}) dec = Scalar{0};
        node->impurity_decrease = dec;

        // Track normalised feature importance contribution (Gini-based).
        // sklearn weights by node sample count / total samples.
        if (total_weight_ > Scalar{0}) {
            feature_importances_(best_feature) +=
                (static_cast<Scalar>(n) / total_weight_) * node->impurity_decrease;
        }

        node->left = build_tree(X, y, best_left, depth + 1);
        node->right = build_tree(X, y, best_right, depth + 1);
        return node;
    }

    template <typename RowExpr>
    static const Node* walk_to_leaf(const Node* node, const RowExpr& x) {
        while (!node->is_leaf) {
            node = (x(node->feature_index) <= node->threshold)
                ? node->left.get() : node->right.get();
        }
        return node;
    }
};

/// @brief A decision tree regressor.
///
/// Uses MSE (Mean Squared Error) reduction as the splitting criterion.
/// The prediction for a leaf node is the mean of the target values.
///
/// Mirrors
/// [sklearn.tree.DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html).
template <typename Scalar = double>
class DecisionTreeRegressor
    : public Predictor<DecisionTreeRegressor<Scalar>, Scalar> {
public:
    using Base = Predictor<DecisionTreeRegressor<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;
    using typename Base::ScalarType;

    explicit DecisionTreeRegressor(int max_depth = -1,
                                   int min_samples_split = 2,
                                   int max_features_mode = 0,
                                   double max_features_value = 0.0,
                                   std::optional<uint64_t> random_state = std::nullopt)
        : max_depth_(max_depth),
          min_samples_split_(min_samples_split),
          max_features_mode_(max_features_mode),
          max_features_value_(max_features_value),
          random_state_(random_state) {}

    /// @brief Build a decision tree regressor from the training set.
    DecisionTreeRegressor& fit_impl(const Eigen::Ref<const MatrixType>& X,
                               const Eigen::Ref<const VectorType>& y) {
        return fit_with_indices(X, y, std::vector<Eigen::Index>{});
    }

    DecisionTreeRegressor& fit_with_indices(const Eigen::Ref<const MatrixType>& X,
                                            const Eigen::Ref<const VectorType>& y,
                                            const std::vector<Eigen::Index>& sample_indices) {
        internal::check_non_empty(X);
        internal::check_consistent_length(X, y);

        this->n_features_in_ = X.cols();

        std::vector<Eigen::Index> indices;
        if (sample_indices.empty()) {
            indices.resize(static_cast<std::size_t>(X.rows()));
            std::iota(indices.begin(), indices.end(), Eigen::Index{0});
        } else {
            indices = sample_indices;
        }

        feature_importances_ = RowVectorType::Zero(X.cols());
        total_weight_ = static_cast<Scalar>(indices.size());

        if (random_state_.has_value()) rng_ = std::mt19937_64(*random_state_);
        else {
            std::random_device rd;
            rng_ = std::mt19937_64(static_cast<uint64_t>(rd()));
        }

        root_ = build_tree(X, y, indices, 0);

        Scalar s = feature_importances_.sum();
        if (s > Scalar{0}) feature_importances_ /= s;

        this->fitted_ = true;
        return *this;
    }

    /// @brief Predict target values for X.
    [[nodiscard]] VectorType predict_impl(
        const Eigen::Ref<const MatrixType>& X) const {

        VectorType predictions(X.rows());
        for (Eigen::Index i = 0; i < X.rows(); ++i) {
            predictions(i) = predict_one(root_.get(), X.row(i));
        }
        return predictions;
    }

    /// @brief Return the @f$R^2@f$ coefficient of determination.
    [[nodiscard]] ScalarType score_impl(const Eigen::Ref<const MatrixType>& X,
                               const Eigen::Ref<const VectorType>& y) const {
        VectorType y_pred = predict_impl(X);
        Scalar ss_res = (y - y_pred).squaredNorm();
        Scalar ss_tot = (y.array() - y.mean()).matrix().squaredNorm();
        if (ss_tot == Scalar{0}) return Scalar{0};
        return Scalar{1} - ss_res / ss_tot;
    }

    [[nodiscard]] const RowVectorType& feature_importances() const {
        this->check_is_fitted(); return feature_importances_;
    }

private:
    int max_depth_;
    int min_samples_split_;
    int max_features_mode_;
    double max_features_value_;
    std::optional<uint64_t> random_state_;

    RowVectorType feature_importances_;
    Scalar total_weight_{0};
    mutable std::mt19937_64 rng_;

    std::unique_ptr<internal::TreeNode<Scalar>> root_;

    using Node = internal::TreeNode<Scalar>;

    std::unique_ptr<Node> build_tree(
        const Eigen::Ref<const MatrixType>& X,
        const Eigen::Ref<const VectorType>& y,
        const std::vector<Eigen::Index>& indices,
        int depth) {

        auto node = std::make_unique<Node>();
        const auto n = static_cast<int>(indices.size());

        // Mean and MSE at this node.
        Scalar mean_val{0};
        for (auto idx : indices) mean_val += y(idx);
        mean_val /= static_cast<Scalar>(n);

        Scalar mse_node{0};
        for (auto idx : indices) {
            Scalar d = y(idx) - mean_val;
            mse_node += d * d;
        }
        mse_node /= static_cast<Scalar>(n);

        node->n_node_samples = static_cast<Scalar>(n);
        node->impurity = mse_node;
        node->prediction_value = mean_val;

        // Leaf conditions.
        if (n < min_samples_split_ ||
            (max_depth_ >= 0 && depth >= max_depth_) ||
            mse_node == Scalar{0}) {
            node->is_leaf = true;
            return node;
        }

        const Eigen::Index n_features = X.cols();
        const int max_feat = internal::compute_max_features_eff(
            max_features_mode_, max_features_value_, static_cast<int>(n_features));

        std::vector<Eigen::Index> feat_pool(static_cast<std::size_t>(n_features));
        std::iota(feat_pool.begin(), feat_pool.end(), Eigen::Index{0});
        for (int i = 0; i < max_feat && i < static_cast<int>(feat_pool.size()); ++i) {
            std::uniform_int_distribution<int> dist(i, static_cast<int>(feat_pool.size()) - 1);
            int j = dist(rng_);
            std::swap(feat_pool[static_cast<std::size_t>(i)], feat_pool[static_cast<std::size_t>(j)]);
        }

        Scalar best_mse = std::numeric_limits<Scalar>::max();
        Eigen::Index best_feature = -1;
        Scalar best_threshold{0};
        std::vector<Eigen::Index> best_left, best_right;
        bool found = false;

        for (int fi = 0; fi < max_feat; ++fi) {
            Eigen::Index f = feat_pool[static_cast<std::size_t>(fi)];
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
                    found = true;
                }
            }
        }

        if (!found) {
            node->is_leaf = true;
            return node;
        }

        node->feature_index = best_feature;
        node->threshold = best_threshold;
        Scalar dec = node->impurity - best_mse;
        if (dec < Scalar{0}) dec = Scalar{0};
        node->impurity_decrease = dec;
        if (total_weight_ > Scalar{0}) {
            feature_importances_(best_feature) +=
                (static_cast<Scalar>(n) / total_weight_) * node->impurity_decrease;
        }

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

/// @}

} // namespace Skigen

#endif // SKIGEN_TREE_DECISION_TREE_H
