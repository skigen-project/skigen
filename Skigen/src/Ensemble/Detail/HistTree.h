// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

#ifndef SKIGEN_ENSEMBLE_DETAIL_HIST_TREE_H
#define SKIGEN_ENSEMBLE_DETAIL_HIST_TREE_H

#include <Eigen/Core>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <queue>
#include <vector>

namespace Skigen {
namespace internal {

/// @brief A single node in a histogram gradient-boosting tree.
///
/// Internal nodes split on `feature`. For an **ordered** (numerical) feature
/// a sample goes left when its binned value is `<= bin_threshold`. For a
/// **categorical** feature (`is_categorical`), a sample goes left when its bin
/// is a member of `cat_left` (a per-bin membership mask). Leaves carry a
/// `value` (the Newton step @f$-G/(H+\lambda)@f$ for the samples they cover).
template <typename Scalar>
struct HistTreeNode {
    bool is_leaf = true;
    int feature = -1;
    uint8_t bin_threshold = 0;
    bool is_categorical = false;
    std::vector<uint8_t> cat_left;  // bin -> goes-left (categorical only)
    int left = -1;
    int right = -1;
    Scalar value = Scalar{0};
};

/// @brief Hyperparameters for the histogram tree grower.
template <typename Scalar>
struct HistTreeParams {
    int max_leaf_nodes = 31;     // <= 0 means unbounded
    int max_depth = -1;          // < 0 means unbounded
    int min_samples_leaf = 20;
    Scalar l2_regularization = Scalar{0};
    int n_bins = 256;            // bins per feature (max bin index + 1)
    Scalar min_gain_to_split = Scalar{0};
    // Per-feature monotonic constraint: +1 increasing, -1 decreasing,
    // 0 none. Empty means no constraints.
    std::vector<int> monotonic_cst;
    // Per-feature categorical flag (1 = categorical / unordered bins).
    // Empty means all features are ordered (numerical).
    std::vector<int> categorical_features;
};

/// @brief A histogram-based, leaf-wise gradient-boosting regression tree.
///
/// Implements the LightGBM / sklearn `HistGradientBoosting` inner tree:
/// gradient/hessian histograms over a pre-binned design matrix, a
/// second-order (Newton) split-gain criterion with L2 regularisation,
/// best-first (leaf-wise) growth bounded by `max_leaf_nodes`, and
/// per-feature monotonic constraints enforced via value bounds.
///
/// The grain is one tree fit per boosting iteration. The binned matrix
/// `X_binned` is (n_samples × n_features) with values in
/// `[0, n_bins - 1]`; the caller supplies per-sample gradients and
/// hessians for whatever loss is being boosted.
///
/// Reference: Ke et al., "LightGBM: A Highly Efficient Gradient Boosting
/// Decision Tree" (NeurIPS 2017); the split gain follows Chen & Guestrin,
/// "XGBoost" (KDD 2016), Eq. 7.
template <typename Scalar>
class HistTree {
public:
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using BinnedMatrix =
        Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>;

    HistTree() = default;

    /// @brief Fit the tree on binned data with given gradients/hessians.
    void fit(const BinnedMatrix& X_binned, const Vector& grad,
             const Vector& hess, const HistTreeParams<Scalar>& params) {
        params_ = params;
        nodes_.clear();
        const Eigen::Index n = X_binned.rows();

        // Root covers all samples.
        std::vector<int> root_idx(static_cast<std::size_t>(n));
        for (Eigen::Index i = 0; i < n; ++i)
            root_idx[static_cast<std::size_t>(i)] = static_cast<int>(i);

        nodes_.emplace_back();  // node 0 = root (leaf for now)
        const int n_leaves_target =
            (params_.max_leaf_nodes > 0) ? params_.max_leaf_nodes
                                         : std::numeric_limits<int>::max();

        // Frontier of growable leaves, ordered by best split gain.
        std::priority_queue<Candidate> frontier;
        Candidate root_cand = evaluate(X_binned, grad, hess, root_idx, 0,
                                       lower_inf(), upper_inf());
        root_cand.node = 0;
        root_cand.samples = std::move(root_idx);
        // Set the root leaf value up front (used if it is never split).
        nodes_[0].value = leaf_value(sum_grad(grad, root_cand.samples),
                                     sum_hess(hess, root_cand.samples));
        if (root_cand.has_split) frontier.push(std::move(root_cand));

        int n_leaves = 1;
        while (!frontier.empty() && n_leaves < n_leaves_target) {
            Candidate c = frontier.top();
            frontier.pop();
            if (!c.has_split) continue;

            // Partition samples by the chosen split (ordered or categorical).
            std::vector<int> left_idx, right_idx;
            left_idx.reserve(c.samples.size());
            right_idx.reserve(c.samples.size());
            for (int s : c.samples) {
                const bool go_left = c.is_categorical
                    ? (c.cat_left[X_binned(s, c.feature)] != 0)
                    : (X_binned(s, c.feature) <= c.bin_threshold);
                if (go_left) left_idx.push_back(s);
                else right_idx.push_back(s);
            }

            // Materialise the split into the node tree.
            const int li = static_cast<int>(nodes_.size());
            const int ri = li + 1;
            nodes_.emplace_back();
            nodes_.emplace_back();

            nodes_[c.node].is_leaf = false;
            nodes_[c.node].feature = c.feature;
            nodes_[c.node].bin_threshold = c.bin_threshold;
            nodes_[c.node].is_categorical = c.is_categorical;
            nodes_[c.node].cat_left = c.cat_left;
            nodes_[c.node].left = li;
            nodes_[c.node].right = ri;

            nodes_[li].value = c.left_value;
            nodes_[ri].value = c.right_value;
            ++n_leaves;

            const int child_depth = c.depth + 1;
            // Evaluate children with the (possibly tightened) value bounds
            // implied by the monotonic constraint at this split.
            Candidate lc = evaluate(X_binned, grad, hess, left_idx,
                                    child_depth, c.left_lower, c.left_upper);
            lc.node = li;
            lc.samples = std::move(left_idx);
            if (lc.has_split) frontier.push(std::move(lc));

            Candidate rc = evaluate(X_binned, grad, hess, right_idx,
                                    child_depth, c.right_lower, c.right_upper);
            rc.node = ri;
            rc.samples = std::move(right_idx);
            if (rc.has_split) frontier.push(std::move(rc));
        }
    }

    /// @brief Predict the raw additive update for each row of `X_binned`.
    [[nodiscard]] Vector predict(const BinnedMatrix& X_binned) const {
        const Eigen::Index n = X_binned.rows();
        Vector out(n);
        for (Eigen::Index i = 0; i < n; ++i) {
            int node = 0;
            while (!nodes_[static_cast<std::size_t>(node)].is_leaf) {
                const auto& nd = nodes_[static_cast<std::size_t>(node)];
                const bool go_left = nd.is_categorical
                    ? (nd.cat_left[X_binned(i, nd.feature)] != 0)
                    : (X_binned(i, nd.feature) <= nd.bin_threshold);
                node = go_left ? nd.left : nd.right;
            }
            out(i) = nodes_[static_cast<std::size_t>(node)].value;
        }
        return out;
    }

    [[nodiscard]] int n_leaves() const noexcept {
        int leaves = 0;
        for (const auto& nd : nodes_)
            if (nd.is_leaf) ++leaves;
        return leaves;
    }

private:
    static Scalar lower_inf() {
        return -std::numeric_limits<Scalar>::infinity();
    }
    static Scalar upper_inf() {
        return std::numeric_limits<Scalar>::infinity();
    }

    // A pending leaf and its best candidate split.
    struct Candidate {
        int node = -1;
        int depth = 0;
        std::vector<int> samples;
        bool has_split = false;
        Scalar gain = -std::numeric_limits<Scalar>::infinity();
        int feature = -1;
        uint8_t bin_threshold = 0;
        bool is_categorical = false;
        std::vector<uint8_t> cat_left;
        Scalar left_value = Scalar{0};
        Scalar right_value = Scalar{0};
        // Value bounds propagated to each child for monotonic constraints.
        Scalar left_lower = -std::numeric_limits<Scalar>::infinity();
        Scalar left_upper = std::numeric_limits<Scalar>::infinity();
        Scalar right_lower = -std::numeric_limits<Scalar>::infinity();
        Scalar right_upper = std::numeric_limits<Scalar>::infinity();

        bool operator<(const Candidate& o) const { return gain < o.gain; }
    };

    Scalar leaf_value(Scalar g, Scalar h) const {
        return -g / (h + params_.l2_regularization);
    }

    static Scalar sum_grad(const Vector& grad, const std::vector<int>& idx) {
        Scalar s{0};
        for (int i : idx) s += grad(i);
        return s;
    }
    static Scalar sum_hess(const Vector& hess, const std::vector<int>& idx) {
        Scalar s{0};
        for (int i : idx) s += hess(i);
        return s;
    }

    Scalar clamp(Scalar v, Scalar lo, Scalar hi) const {
        return std::min(std::max(v, lo), hi);
    }

    // Find the best split for a leaf covering `idx`, honouring depth,
    // min_samples_leaf, L2, monotonic constraints, and the value bounds
    // [lower, upper] inherited from ancestors.
    Candidate evaluate(const BinnedMatrix& X_binned, const Vector& grad,
                       const Vector& hess, const std::vector<int>& idx,
                       int depth, Scalar lower, Scalar upper) const {
        Candidate c;
        c.depth = depth;
        const int n_node = static_cast<int>(idx.size());

        const Scalar G = sum_grad(grad, idx);
        const Scalar H = sum_hess(hess, idx);

        // Depth / size stopping conditions: leaf stays a leaf.
        if (params_.max_depth >= 0 && depth >= params_.max_depth) return c;
        if (n_node < 2 * params_.min_samples_leaf) return c;

        const Scalar lambda = params_.l2_regularization;
        const Scalar parent_obj = (G * G) / (H + lambda);
        const Eigen::Index p = X_binned.cols();
        const bool has_mono = !params_.monotonic_cst.empty();
        const bool has_cat = !params_.categorical_features.empty();

        for (Eigen::Index f = 0; f < p; ++f) {
            // Build gradient/hessian histograms over the bins of feature f.
            std::vector<Scalar> hist_g(
                static_cast<std::size_t>(params_.n_bins), Scalar{0});
            std::vector<Scalar> hist_h(
                static_cast<std::size_t>(params_.n_bins), Scalar{0});
            std::vector<int> hist_c(
                static_cast<std::size_t>(params_.n_bins), 0);
            for (int s : idx) {
                const uint8_t b = X_binned(s, f);
                hist_g[b] += grad(s);
                hist_h[b] += hess(s);
                hist_c[b] += 1;
            }

            const int mono = has_mono
                ? params_.monotonic_cst[static_cast<std::size_t>(f)]
                : 0;
            const bool is_cat =
                has_cat &&
                params_.categorical_features[static_cast<std::size_t>(f)] != 0;

            if (is_cat) {
                // Categorical (unordered) split: sort the present bins by
                // their gradient/hessian ratio, then scan the prefix as the
                // left child (LightGBM / sklearn's one-hot-free strategy).
                // Monotonic constraints do not apply to categoricals.
                std::vector<int> bins;
                for (int b = 0; b < params_.n_bins; ++b)
                    if (hist_c[static_cast<std::size_t>(b)] > 0) bins.push_back(b);
                std::sort(bins.begin(), bins.end(), [&](int a, int b) {
                    const Scalar ra = hist_g[static_cast<std::size_t>(a)] /
                        (hist_h[static_cast<std::size_t>(a)] + lambda);
                    const Scalar rb = hist_g[static_cast<std::size_t>(b)] /
                        (hist_h[static_cast<std::size_t>(b)] + lambda);
                    return ra < rb;
                });

                Scalar gl{0}, hl{0};
                int cl = 0;
                for (std::size_t t = 0; t + 1 < bins.size(); ++t) {
                    const int b = bins[t];
                    gl += hist_g[static_cast<std::size_t>(b)];
                    hl += hist_h[static_cast<std::size_t>(b)];
                    cl += hist_c[static_cast<std::size_t>(b)];
                    if (cl < params_.min_samples_leaf) continue;
                    const int cr = n_node - cl;
                    if (cr < params_.min_samples_leaf) break;

                    const Scalar gr = G - gl;
                    const Scalar hr = H - hl;
                    const Scalar gain =
                        (gl * gl) / (hl + lambda) +
                        (gr * gr) / (hr + lambda) - parent_obj;
                    if (gain <= params_.min_gain_to_split || gain <= c.gain)
                        continue;

                    c.has_split = true;
                    c.gain = gain;
                    c.feature = static_cast<int>(f);
                    c.is_categorical = true;
                    c.cat_left.assign(
                        static_cast<std::size_t>(params_.n_bins), 0);
                    for (std::size_t u = 0; u <= t; ++u)
                        c.cat_left[static_cast<std::size_t>(bins[u])] = 1;
                    c.left_value = clamp(leaf_value(gl, hl), lower, upper);
                    c.right_value = clamp(leaf_value(gr, hr), lower, upper);
                    c.left_lower = lower;  c.left_upper = upper;
                    c.right_lower = lower; c.right_upper = upper;
                }
                continue;
            }

            // Sweep bins left to right, accumulating the left child.
            Scalar gl{0}, hl{0};
            int cl = 0;
            for (int b = 0; b < params_.n_bins - 1; ++b) {
                gl += hist_g[static_cast<std::size_t>(b)];
                hl += hist_h[static_cast<std::size_t>(b)];
                cl += hist_c[static_cast<std::size_t>(b)];
                if (cl < params_.min_samples_leaf) continue;
                const int cr = n_node - cl;
                if (cr < params_.min_samples_leaf) break;

                const Scalar gr = G - gl;
                const Scalar hr = H - hl;
                const Scalar gain =
                    (gl * gl) / (hl + lambda) +
                    (gr * gr) / (hr + lambda) - parent_obj;
                if (gain <= params_.min_gain_to_split || gain <= c.gain)
                    continue;

                Scalar vl = clamp(leaf_value(gl, hl), lower, upper);
                Scalar vr = clamp(leaf_value(gr, hr), lower, upper);

                // Monotonic constraint: enforce ordering between children.
                if (mono != 0) {
                    if (mono > 0 && vl > vr) continue;  // must increase
                    if (mono < 0 && vl < vr) continue;  // must decrease
                }

                c.has_split = true;
                c.gain = gain;
                c.feature = static_cast<int>(f);
                c.is_categorical = false;
                c.cat_left.clear();
                c.bin_threshold = static_cast<uint8_t>(b);
                c.left_value = vl;
                c.right_value = vr;

                // Propagate child value bounds for the monotonic case so
                // descendants cannot violate the order across this split.
                c.left_lower = lower;
                c.left_upper = upper;
                c.right_lower = lower;
                c.right_upper = upper;
                if (mono > 0) {
                    const Scalar mid = (vl + vr) / Scalar{2};
                    c.left_upper = std::min(upper, mid);
                    c.right_lower = std::max(lower, mid);
                } else if (mono < 0) {
                    const Scalar mid = (vl + vr) / Scalar{2};
                    c.left_lower = std::max(lower, mid);
                    c.right_upper = std::min(upper, mid);
                }
            }
        }
        return c;
    }

    HistTreeParams<Scalar> params_;
    std::vector<HistTreeNode<Scalar>> nodes_;
};

}  // namespace internal
}  // namespace Skigen

#endif  // SKIGEN_ENSEMBLE_DETAIL_HIST_TREE_H
