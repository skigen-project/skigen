// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

#ifndef SKIGEN_MANIFOLD_DETAIL_QUADTREE_H
#define SKIGEN_MANIFOLD_DETAIL_QUADTREE_H

#include <Eigen/Core>

#include <array>
#include <cmath>
#include <vector>

namespace Skigen {
namespace internal {

/// @brief Barnes-Hut quadtree for 2-D t-SNE repulsive-force approximation.
///
/// Builds a quadtree over the current embedding and, for a query point,
/// approximates the Student-t repulsive forces and the normalisation
/// constant @f$Z = \sum_{k \ne l} (1 + \lVert y_k - y_l \rVert^2)^{-1}@f$.
/// A tree cell is treated as a single summary point when the Barnes-Hut
/// criterion @f$ (r_\text{cell} / d) < \theta @f$ holds, where
/// @f$r_\text{cell}@f$ is the cell width and @f$d@f$ the distance to the
/// cell's centre of mass. This reduces the repulsive sum from
/// @f$O(n^2)@f$ to @f$O(n \log n)@f$.
///
/// Reference: van der Maaten, "Accelerating t-SNE using Tree-Based
/// Algorithms" (JMLR 2014), Algorithm 2.
template <typename Scalar>
class QuadTree {
public:
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    /// @brief Build the tree over the (n × 2) embedding `Y`.
    explicit QuadTree(const Matrix& Y) {
        const Eigen::Index n = Y.rows();
        // Bounding box of all points.
        Scalar min_x = Y(0, 0), max_x = Y(0, 0);
        Scalar min_y = Y(0, 1), max_y = Y(0, 1);
        for (Eigen::Index i = 1; i < n; ++i) {
            min_x = std::min(min_x, Y(i, 0));
            max_x = std::max(max_x, Y(i, 0));
            min_y = std::min(min_y, Y(i, 1));
            max_y = std::max(max_y, Y(i, 1));
        }
        const Scalar cx = (min_x + max_x) / Scalar{2};
        const Scalar cy = (min_y + max_y) / Scalar{2};
        const Scalar half =
            std::max(max_x - min_x, max_y - min_y) / Scalar{2} + Scalar{1e-6};

        nodes_.reserve(static_cast<std::size_t>(2 * n + 1));
        nodes_.emplace_back(cx, cy, half);
        for (Eigen::Index i = 0; i < n; ++i)
            insert(0, Y(i, 0), Y(i, 1));
    }

    /// @brief Accumulate the negative (repulsive) force on point `(px, py)`
    ///   and add the corresponding contribution to the normaliser `Z`.
    ///
    /// On return `fx`, `fy` hold the unnormalised repulsive force
    /// @f$\sum_k q_{lk}^2 Z (y_l - y_k)@f$ accumulators (in the form
    /// @f$\sum_k (1 + d^2)^{-2}(y_l - y_k)@f$), and `Z` is incremented by
    /// @f$\sum_k (1 + d^2)^{-1}@f$.
    void compute_force(Scalar px, Scalar py, Scalar theta, Scalar& fx,
                       Scalar& fy, Scalar& Z) const {
        accumulate(0, px, py, theta * theta, fx, fy, Z);
    }

private:
    struct Node {
        Scalar cx, cy, half;          // cell centre and half-width
        Scalar mass = Scalar{0};      // number of points in subtree
        Scalar com_x = Scalar{0};     // centre of mass
        Scalar com_y = Scalar{0};
        bool is_leaf = true;
        bool has_point = false;
        Scalar px = Scalar{0}, py = Scalar{0};  // the single leaf point
        std::array<int, 4> child{{-1, -1, -1, -1}};
        Node(Scalar x, Scalar y, Scalar h) : cx(x), cy(y), half(h) {}
    };

    int quadrant(const Node& nd, Scalar x, Scalar y) const {
        int q = 0;
        if (x > nd.cx) q += 1;
        if (y > nd.cy) q += 2;
        return q;
    }

    int make_child(const Node& parent, int q) {
        const Scalar h = parent.half / Scalar{2};
        const Scalar ox = (q & 1) ? h : -h;
        const Scalar oy = (q & 2) ? h : -h;
        nodes_.emplace_back(parent.cx + ox, parent.cy + oy, h);
        return static_cast<int>(nodes_.size()) - 1;
    }

    void insert(int idx, Scalar x, Scalar y) {
        // Update centre of mass incrementally.
        {
            Node& nd = nodes_[static_cast<std::size_t>(idx)];
            nd.com_x = (nd.com_x * nd.mass + x) / (nd.mass + Scalar{1});
            nd.com_y = (nd.com_y * nd.mass + y) / (nd.mass + Scalar{1});
            nd.mass += Scalar{1};
        }

        if (nodes_[static_cast<std::size_t>(idx)].is_leaf) {
            if (!nodes_[static_cast<std::size_t>(idx)].has_point) {
                nodes_[static_cast<std::size_t>(idx)].has_point = true;
                nodes_[static_cast<std::size_t>(idx)].px = x;
                nodes_[static_cast<std::size_t>(idx)].py = y;
                return;
            }
            // Subdivide: re-insert the existing point, then the new one.
            const Scalar ox = nodes_[static_cast<std::size_t>(idx)].px;
            const Scalar oy = nodes_[static_cast<std::size_t>(idx)].py;
            nodes_[static_cast<std::size_t>(idx)].is_leaf = false;
            nodes_[static_cast<std::size_t>(idx)].has_point = false;
            reinsert_into_child(idx, ox, oy);
            reinsert_into_child(idx, x, y);
            return;
        }
        reinsert_into_child(idx, x, y);
    }

    void reinsert_into_child(int idx, Scalar x, Scalar y) {
        const int q = quadrant(nodes_[static_cast<std::size_t>(idx)], x, y);
        if (nodes_[static_cast<std::size_t>(idx)].child[static_cast<std::size_t>(q)] < 0) {
            const int c = make_child(nodes_[static_cast<std::size_t>(idx)], q);
            nodes_[static_cast<std::size_t>(idx)].child[static_cast<std::size_t>(q)] = c;
        }
        insert(nodes_[static_cast<std::size_t>(idx)].child[static_cast<std::size_t>(q)],
               x, y);
    }

    void accumulate(int idx, Scalar px, Scalar py, Scalar theta_sq,
                    Scalar& fx, Scalar& fy, Scalar& Z) const {
        const Node& nd = nodes_[static_cast<std::size_t>(idx)];
        if (nd.mass == Scalar{0}) return;

        const Scalar dx = px - nd.com_x;
        const Scalar dy = py - nd.com_y;
        const Scalar dist_sq = dx * dx + dy * dy;

        // Barnes-Hut criterion: (cell_width^2 / dist^2) < theta^2.
        const Scalar width = Scalar{2} * nd.half;
        if (nd.is_leaf || (width * width < theta_sq * dist_sq)) {
            if (dist_sq < Scalar{1e-12}) return;  // skip self / coincident
            const Scalar qz = Scalar{1} / (Scalar{1} + dist_sq);
            Z += nd.mass * qz;
            const Scalar mult = nd.mass * qz * qz;
            fx += mult * dx;
            fy += mult * dy;
            return;
        }
        for (int q = 0; q < 4; ++q)
            if (nd.child[static_cast<std::size_t>(q)] >= 0)
                accumulate(nd.child[static_cast<std::size_t>(q)], px, py,
                           theta_sq, fx, fy, Z);
    }

    std::vector<Node> nodes_;
};

}  // namespace internal
}  // namespace Skigen

#endif  // SKIGEN_MANIFOLD_DETAIL_QUADTREE_H
