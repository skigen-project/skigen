// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_MANIFOLD_ISOMAP_H
#define SKIGEN_MANIFOLD_ISOMAP_H

#include "../Core/Base.h"
#include "../Core/Validation.h"
#include "Detail/TruncatedEigensolver.h"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <string>
#include <utility>
#include <vector>

namespace Skigen {

/// @defgroup Algo_Isomap Isomap
/// @ingroup Manifold
/// @brief Isometric Mapping (Isomap) via geodesic distances and classical MDS.
/// @{

/// @brief Isometric Mapping (Isomap).
///
/// Non-linear dimensionality reduction that preserves geodesic
/// distances between all pairs of points.  The algorithm builds a
/// K-nearest-neighbours graph, computes shortest-path (geodesic)
/// distances with Dijkstra's algorithm, then applies classical
/// (metric) MDS via eigendecomposition of the double-centred squared
/// distance matrix.
///
/// Mirrors
/// [sklearn.manifold.Isomap](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html).
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `n_components` | `int` | `2` | Dimensionality of the embedding. |
/// | `n_neighbors` | `int` | `5` | Number of neighbours for the KNN graph. |
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type | Description |
/// |----------|------|-------------|
/// | `embedding()` | `MatrixType` | Embedding coordinates (n_samples x n_components). |
/// | `dist_matrix()` | `MatrixType` | Geodesic distance matrix (n_samples x n_samples). |
///
/// ### Limitations relative to scikit-learn
///
/// The parameters `radius`, `tol`, `max_iter`, `path_method`,
/// `neighbors_algorithm`, `n_jobs`, `p`, and `metric` are not supported.
/// `eigen_solver` accepts `"auto"`, `"arpack"`, and `"dense"` (the
/// `"arpack"` truncated path requires `SKIGEN_ENABLE_SPECTRA`).
///
/// ### Examples
///
/// @snippet isomap.cpp example_isomap
template <typename Scalar = double>
class Isomap
    : public Transformer<Isomap<Scalar>, Scalar> {
public:
    using Base = Transformer<Isomap<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;

    /// @brief Construct an Isomap estimator.
    ///
    /// @param n_components Embedding dimensionality (default 2).
    /// @param n_neighbors  Number of nearest neighbours for the graph (default 5).
    /// @param eigen_solver Eigensolver backend: `"auto"`, `"arpack"`, or
    ///   `"dense"` (default `"auto"`). `"arpack"` is only active under
    ///   `SKIGEN_ENABLE_SPECTRA`; otherwise the dense solver is used.
    explicit Isomap(int n_components = 2, int n_neighbors = 5,
                    std::string eigen_solver = "auto")
        : n_components_(n_components),
          n_neighbors_(n_neighbors),
          eigen_solver_(std::move(eigen_solver)) {}

    // -- Accessors ----------------------------------------------------------

    /// @brief Embedding coordinates (n_samples x n_components).
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] const MatrixType& embedding() const {
        this->check_is_fitted(); return embedding_;
    }
    /// @brief Geodesic (shortest-path) distance matrix (n_samples x n_samples).
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] const MatrixType& dist_matrix() const {
        this->check_is_fitted(); return dist_matrix_;
    }
    /// @brief The configured eigensolver name (`"auto"` / `"arpack"` / `"dense"`).
    [[nodiscard]] const std::string& eigen_solver() const noexcept {
        return eigen_solver_;
    }

    SKIGEN_PARAMS(
        (n_components, n_components_, int),
        (n_neighbors, n_neighbors_, int),
        (eigen_solver, eigen_solver_, std::string))

    // -- Implementation (called by CRTP base) --------------------------------

    /// @brief Fit the Isomap model.
    ///
    /// Builds a KNN graph from the Euclidean distances of X, computes
    /// all-pairs shortest paths via Dijkstra, then applies classical MDS
    /// (double-centring + eigendecomposition) to obtain the embedding.
    ///
    /// @param X Data matrix of shape (n_samples, n_features).
    /// @return Reference to the fitted transformer (`*this`).
    Isomap& fit_impl(const Eigen::Ref<const MatrixType>& X) {
        internal::check_non_empty(X);

        this->n_features_in_ = X.cols();
        const Eigen::Index n = X.rows();
        const int k = std::min(n_neighbors_,
                               static_cast<int>(n - 1));

        // 1. Pairwise Euclidean distances.
        MatrixType D = pairwise_distances(X);

        // 2. Build KNN adjacency matrix (symmetric).
        //    adj(i,j) = Euclidean distance if j is among i's k-nearest
        //    neighbours (or vice-versa), infinity otherwise.
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        MatrixType adj = MatrixType::Constant(n, n, inf);
        for (Eigen::Index i = 0; i < n; ++i) adj(i, i) = Scalar{0};

        for (Eigen::Index i = 0; i < n; ++i) {
            // Collect (distance, index) pairs for row i.
            std::vector<std::pair<Scalar, Eigen::Index>> nbrs;
            nbrs.reserve(n - 1);
            for (Eigen::Index j = 0; j < n; ++j) {
                if (j == i) continue;
                nbrs.emplace_back(D(i, j), j);
            }
            // Partial sort to find the k nearest.
            std::partial_sort(nbrs.begin(),
                              nbrs.begin() + k,
                              nbrs.end());
            for (int m = 0; m < k; ++m) {
                Eigen::Index j = nbrs[m].second;
                Scalar d = nbrs[m].first;
                adj(i, j) = d;
                adj(j, i) = d;   // Symmetrise.
            }
        }

        // 3. All-pairs shortest paths via Dijkstra.
        dist_matrix_ = all_pairs_dijkstra(adj, n);

        // Handle disconnected components: replace infinite distances with
        // twice the maximum finite distance so the eigendecomposition
        // remains well-conditioned.
        Scalar max_finite = Scalar{0};
        for (Eigen::Index i = 0; i < n; ++i)
            for (Eigen::Index j = 0; j < n; ++j)
                if (std::isfinite(dist_matrix_(i, j)))
                    max_finite = std::max(max_finite, dist_matrix_(i, j));

        const Scalar fill = max_finite * Scalar{2};
        for (Eigen::Index i = 0; i < n; ++i)
            for (Eigen::Index j = 0; j < n; ++j)
                if (!std::isfinite(dist_matrix_(i, j)))
                    dist_matrix_(i, j) = fill;

        // 4. Classical MDS via double-centring.
        //    B = -1/2 * H * D^2 * H,   H = I - 11^T / n.
        MatrixType D2 = dist_matrix_.array().square().matrix();
        MatrixType H = MatrixType::Identity(n, n)
                       - MatrixType::Ones(n, n) / static_cast<Scalar>(n);
        MatrixType B = Scalar{-0.5} * H * D2 * H;

        // Classical MDS keeps the largest eigenpairs of B. Route through the
        // shared truncated-eigensolver helper (dense by default; Spectra
        // ARPACK-style only under SKIGEN_ENABLE_SPECTRA).
        const internal::EigenSolver solver =
            internal::parse_eigen_solver(eigen_solver_);
        const Eigen::Index nc = std::min(static_cast<Eigen::Index>(n_components_), n);
        VectorType vals;
        MatrixType vecs;
        internal::largest_eigenpairs<Scalar>(B, static_cast<int>(nc), solver,
                                             vals, vecs);

        // Clamp tiny negative eigenvalues arising from numerical error.
        vals = vals.array().max(Scalar{0});

        // Embedding = eigenvectors * diag(sqrt(eigenvalues)).
        embedding_ = vecs * vals.array().sqrt().matrix().asDiagonal();

        this->fitted_ = true;
        return *this;
    }

    /// @brief Return the stored embedding.
    ///
    /// For Isomap the transform simply returns the embedding computed
    /// during fit.  The input X is expected to match the training data.
    ///
    /// @param X Data matrix (ignored; the stored embedding is returned).
    /// @return The embedding matrix (n_samples x n_components).
    [[nodiscard]] MatrixType transform_impl(
        const Eigen::Ref<const MatrixType>& /*X*/) const {
        return embedding_;
    }

private:
    int n_components_;
    int n_neighbors_;
    std::string eigen_solver_ = "auto";

    MatrixType embedding_;
    MatrixType dist_matrix_;

    /// @brief Compute the pairwise Euclidean distance matrix.
    static MatrixType pairwise_distances(const Eigen::Ref<const MatrixType>& M) {
        const Eigen::Index n = M.rows();
        VectorType sq = M.rowwise().squaredNorm();
        MatrixType D2 = sq.replicate(1, n) + sq.transpose().replicate(n, 1)
                        - Scalar{2} * (M * M.transpose());
        return D2.array().max(Scalar{0}).sqrt().matrix();
    }

    /// @brief Dijkstra's shortest-path algorithm from a single source.
    ///
    /// @param adj  Symmetric adjacency matrix (infinity for no edge).
    /// @param src  Source vertex index.
    /// @param n    Number of vertices.
    /// @return     Vector of shortest distances from src to every vertex.
    static VectorType dijkstra(const MatrixType& adj,
                               Eigen::Index src,
                               Eigen::Index n) {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        VectorType dist = VectorType::Constant(n, inf);
        dist(src) = Scalar{0};

        // Min-heap of (distance, vertex).
        using Entry = std::pair<Scalar, Eigen::Index>;
        std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> pq;
        pq.emplace(Scalar{0}, src);

        while (!pq.empty()) {
            auto [d, u] = pq.top();
            pq.pop();
            if (d > dist(u)) continue;   // Stale entry.

            for (Eigen::Index v = 0; v < n; ++v) {
                if (adj(u, v) == inf) continue;
                Scalar nd = d + adj(u, v);
                if (nd < dist(v)) {
                    dist(v) = nd;
                    pq.emplace(nd, v);
                }
            }
        }
        return dist;
    }

    /// @brief All-pairs shortest paths via repeated Dijkstra.
    static MatrixType all_pairs_dijkstra(const MatrixType& adj,
                                         Eigen::Index n) {
        MatrixType result(n, n);
        for (Eigen::Index i = 0; i < n; ++i)
            result.row(i) = dijkstra(adj, i, n).transpose();
        return result;
    }
};

/// @}

} // namespace Skigen

#endif // SKIGEN_MANIFOLD_ISOMAP_H
