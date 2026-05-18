// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_MANIFOLD_UMAP_H
#define SKIGEN_MANIFOLD_UMAP_H

#include "../Core/Base.h"
#include "../Core/Validation.h"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <vector>

namespace Skigen {

/// @addtogroup Algo_Manifold
/// @{

/// @brief Uniform Manifold Approximation and Projection (UMAP).
///
/// Non-linear dimensionality reduction that preserves both local and
/// global structure.  Constructs a weighted KNN graph in high-dimensional
/// space, then optimises a low-dimensional layout via stochastic gradient
/// descent on a cross-entropy objective.
///
/// Mirrors
/// [umap-learn](https://umap-learn.readthedocs.io/en/latest/).
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `n_components` | `int` | `2` | Dimension of the target embedding. |
/// | `n_neighbors`  | `int` | `15` | Size of the local neighborhood for manifold approximation. |
/// | `min_dist`     | `Scalar` | `0.1` | Minimum distance between embedded points. |
/// | `learning_rate`| `Scalar` | `1.0` | Initial SGD learning rate. |
/// | `n_epochs`     | `int` | `200` | Number of SGD epochs. |
/// | `negative_sample_rate` | `int` | `5` | Negative samples per positive edge. |
/// | `random_state` | `std::optional<uint64_t>` | `nullopt` | Seed for reproducibility. |
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type | Description |
/// |----------|------|-------------|
/// | `embedding()` | `MatrixType` | Low-dimensional embedding (n_samples x n_components). |
///
/// ### Algorithm outline
///
/// 1. Build a KNN graph (brute-force Euclidean).
/// 2. Compute smooth distances @f$\sigma_i@f$ via binary search so that
///    the local connectivity matches @f$\log_2(k)@f$.
/// 3. Construct a symmetrised fuzzy simplicial set from the local
///    membership strengths.
/// 4. Initialise the embedding (spectral or random).
/// 5. Optimise the layout with SGD on the fuzzy cross-entropy loss,
///    applying attractive forces on graph edges and repulsive forces
///    on negative samples.
template <typename Scalar = double>
class UMAP
    : public Transformer<UMAP<Scalar>, Scalar> {
public:
    using Base = Transformer<UMAP<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;

    /// @brief Construct a UMAP estimator.
    ///
    /// @param n_components  Embedding dimension (default 2).
    /// @param n_neighbors   Local neighborhood size (default 15).
    /// @param min_dist      Minimum distance in the embedding (default 0.1).
    /// @param learning_rate Initial SGD learning rate (default 1.0).
    /// @param n_epochs      Number of optimisation epochs (default 200).
    /// @param negative_sample_rate Negative samples per positive edge
    ///   (default 5).
    /// @param random_state  Optional RNG seed (default nullopt).
    explicit UMAP(int n_components = 2,
                  int n_neighbors = 15,
                  Scalar min_dist = Scalar{0.1},
                  Scalar learning_rate = Scalar{1},
                  int n_epochs = 200,
                  int negative_sample_rate = 5,
                  std::optional<uint64_t> random_state = std::nullopt)
        : n_components_(n_components),
          n_neighbors_(n_neighbors),
          min_dist_(min_dist),
          learning_rate_(learning_rate),
          n_epochs_(n_epochs),
          negative_sample_rate_(negative_sample_rate),
          random_state_(random_state) {}

    // -- Accessors ----------------------------------------------------------

    /// @brief Low-dimensional embedding (n_samples x n_components).
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] const MatrixType& embedding() const {
        this->check_is_fitted();
        return embedding_;
    }

    SKIGEN_PARAMS((n_components, n_components_, int),
                  (n_neighbors, n_neighbors_, int),
                  (min_dist, min_dist_, double),
                  (learning_rate, learning_rate_, double),
                  (n_epochs, n_epochs_, int),
                  (negative_sample_rate, negative_sample_rate_, int))

    // -- Implementation (called by CRTP base) --------------------------------

    /// @brief Fit the UMAP model to training data X.
    ///
    /// Builds the fuzzy KNN graph, computes membership strengths, then
    /// runs SGD to optimise the low-dimensional layout.
    ///
    /// @param X Training data of shape (n_samples, n_features).
    /// @return Reference to the fitted transformer (`*this`).
    UMAP& fit_impl(const Eigen::Ref<const MatrixType>& X) {
        internal::check_non_empty(X);

        const Eigen::Index n = X.rows();
        this->n_features_in_ = X.cols();
        const int k = std::min(n_neighbors_,
                               static_cast<int>(n - 1));

        // Seed the RNG
        std::mt19937_64 rng(random_state_.value_or(
            std::random_device{}()));

        // -- Pairwise distance matrix --------------------------------------
        MatrixType dist(n, n);
        for (Eigen::Index i = 0; i < n; ++i) {
            dist(i, i) = Scalar{0};
            for (Eigen::Index j = i + 1; j < n; ++j) {
                Scalar d = (X.row(i) - X.row(j)).norm();
                dist(i, j) = d;
                dist(j, i) = d;
            }
        }

        // -- KNN graph (brute-force) ---------------------------------------
        // knn_idx[i] = sorted indices of k nearest neighbors of i
        // knn_dist[i] = corresponding distances
        std::vector<std::vector<Eigen::Index>> knn_idx(n);
        std::vector<std::vector<Scalar>> knn_dist(n);
        for (Eigen::Index i = 0; i < n; ++i) {
            std::vector<Eigen::Index> idx(n);
            std::iota(idx.begin(), idx.end(), 0);
            std::nth_element(idx.begin(), idx.begin() + k + 1, idx.end(),
                             [&](Eigen::Index a, Eigen::Index b) {
                                 return dist(i, a) < dist(i, b);
                             });
            knn_idx[i].reserve(k);
            knn_dist[i].reserve(k);
            for (int t = 0; t <= k; ++t) {
                if (idx[t] != i) {
                    knn_idx[i].push_back(idx[t]);
                    knn_dist[i].push_back(dist(i, idx[t]));
                }
            }
            if (static_cast<int>(knn_idx[i].size()) > k) {
                knn_idx[i].resize(k);
                knn_dist[i].resize(k);
            }
            // Sort neighbors by distance
            std::vector<std::size_t> order(knn_idx[i].size());
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(),
                      [&](std::size_t a, std::size_t b) {
                          return knn_dist[i][a] < knn_dist[i][b];
                      });
            std::vector<Eigen::Index> sorted_idx(knn_idx[i].size());
            std::vector<Scalar> sorted_dist(knn_dist[i].size());
            for (std::size_t t = 0; t < order.size(); ++t) {
                sorted_idx[t] = knn_idx[i][order[t]];
                sorted_dist[t] = knn_dist[i][order[t]];
            }
            knn_idx[i] = std::move(sorted_idx);
            knn_dist[i] = std::move(sorted_dist);
        }

        // -- Compute smooth distances sigma_i via binary search ------------
        // For each point i, find sigma_i such that
        //   sum_{j in KNN(i)} exp(-max(0, d(i,j) - rho_i) / sigma_i) = log2(k)
        // where rho_i = distance to nearest neighbor.
        const Scalar target = std::log2(static_cast<Scalar>(k));
        std::vector<Scalar> rho(n);
        std::vector<Scalar> sigma(n);

        for (Eigen::Index i = 0; i < n; ++i) {
            rho[i] = knn_dist[i].empty() ? Scalar{0} : knn_dist[i][0];

            Scalar lo = Scalar{1e-8};
            Scalar hi = Scalar{1e4};
            Scalar mid = Scalar{1};

            // Binary search for sigma_i
            for (int iter = 0; iter < 64; ++iter) {
                mid = (lo + hi) / Scalar{2};
                Scalar s = Scalar{0};
                for (auto d : knn_dist[i]) {
                    Scalar shifted = std::max(Scalar{0}, d - rho[i]);
                    s += std::exp(-shifted / mid);
                }
                if (std::abs(s - target) < Scalar{1e-5}) break;
                if (s > target) lo = mid; else hi = mid;
            }
            sigma[i] = mid;
        }

        // -- Construct high-dimensional membership strengths ---------------
        // w(i,j) = exp(-max(0, d(i,j) - rho_i) / sigma_i) for j in KNN(i)
        // Symmetrize: P(i,j) = w(i,j) + w(j,i) - w(i,j)*w(j,i)
        MatrixType W = MatrixType::Zero(n, n);
        for (Eigen::Index i = 0; i < n; ++i) {
            for (std::size_t t = 0; t < knn_idx[i].size(); ++t) {
                Eigen::Index j = knn_idx[i][t];
                Scalar shifted = std::max(Scalar{0},
                                          knn_dist[i][t] - rho[i]);
                W(i, j) = std::exp(-shifted / sigma[i]);
            }
        }

        MatrixType P(n, n);
        for (Eigen::Index i = 0; i < n; ++i) {
            P(i, i) = Scalar{0};
            for (Eigen::Index j = i + 1; j < n; ++j) {
                Scalar wij = W(i, j);
                Scalar wji = W(j, i);
                Scalar p = wij + wji - wij * wji;
                P(i, j) = p;
                P(j, i) = p;
            }
        }

        // -- Collect positive edges with weights ---------------------------
        struct Edge { Eigen::Index i, j; Scalar w; };
        std::vector<Edge> edges;
        Scalar max_w = Scalar{0};
        for (Eigen::Index i = 0; i < n; ++i) {
            for (Eigen::Index j = i + 1; j < n; ++j) {
                if (P(i, j) > Scalar{0}) {
                    edges.push_back({i, j, P(i, j)});
                    max_w = std::max(max_w, P(i, j));
                }
            }
        }
        // Normalise edge weights to [0,1] for epoch sampling
        if (max_w > Scalar{0}) {
            for (auto& e : edges) e.w /= max_w;
        }

        // -- Compute a, b from min_dist -----------------------------------
        // Simplified: b = 1, a = 1/min_dist (or 1e8 when min_dist ~ 0)
        Scalar a_param, b_param;
        b_param = Scalar{1};
        if (min_dist_ > Scalar{1e-6}) {
            a_param = Scalar{1} / min_dist_;
        } else {
            a_param = Scalar{1e8};
        }

        // -- Initialise embedding ------------------------------------------
        // Try spectral initialisation; fall back to random if it fails.
        embedding_ = MatrixType::Zero(n, n_components_);
        if (!try_spectral_init(P, n, rng)) {
            std::normal_distribution<Scalar> normal(Scalar{0}, Scalar{1e-4});
            for (Eigen::Index i = 0; i < n; ++i) {
                for (int d = 0; d < n_components_; ++d) {
                    embedding_(i, d) = normal(rng);
                }
            }
        }

        // -- SGD optimisation of cross-entropy loss ------------------------
        const Scalar eps = Scalar{1e-6};

        for (int epoch = 0; epoch < n_epochs_; ++epoch) {
            Scalar lr = learning_rate_ *
                        (Scalar{1} - static_cast<Scalar>(epoch) /
                                         static_cast<Scalar>(n_epochs_));

            for (const auto& edge : edges) {
                // Sample this edge with probability proportional to weight
                if (edge.w < Scalar{1}) {
                    std::uniform_real_distribution<Scalar> u01(Scalar{0},
                                                              Scalar{1});
                    if (u01(rng) > edge.w) continue;
                }

                Eigen::Index i = edge.i;
                Eigen::Index j = edge.j;

                // -- Attractive force on positive edge (i, j) --------------
                auto diff = embedding_.row(i) - embedding_.row(j);
                Scalar d_sq = diff.squaredNorm();
                Scalar grad_coeff = Scalar{-2} * a_param * b_param *
                                    std::pow(d_sq, b_param - Scalar{1}) /
                                    (Scalar{1} + a_param *
                                         std::pow(d_sq, b_param));

                embedding_.row(i) += lr * grad_coeff * diff;
                embedding_.row(j) -= lr * grad_coeff * diff;

                // -- Repulsive forces on negative samples ------------------
                std::uniform_int_distribution<Eigen::Index> unif(0, n - 1);
                for (int neg = 0; neg < negative_sample_rate_; ++neg) {
                    Eigen::Index kk = unif(rng);
                    if (kk == i) continue;

                    auto diff_neg = embedding_.row(i) - embedding_.row(kk);
                    Scalar d_sq_neg = diff_neg.squaredNorm();

                    Scalar rep_coeff = Scalar{2} * b_param /
                                       (eps + d_sq_neg) /
                                       (Scalar{1} + a_param *
                                            std::pow(d_sq_neg, b_param));

                    embedding_.row(i) += lr * rep_coeff * diff_neg;
                }
            }
        }

        this->fitted_ = true;
        return *this;
    }

    /// @brief Return the stored embedding for the training data.
    ///
    /// @param X Data matrix of shape (n_samples, n_features).
    /// @return Embedding of shape (n_samples, n_components).
    [[nodiscard]] MatrixType transform_impl(
        const Eigen::Ref<const MatrixType>& /*X*/) const {
        return embedding_;
    }

private:
    int n_components_;
    int n_neighbors_;
    Scalar min_dist_;
    Scalar learning_rate_;
    int n_epochs_;
    int negative_sample_rate_;
    std::optional<uint64_t> random_state_;

    MatrixType embedding_;

    // -- Helpers -----------------------------------------------------------

    /// @brief Attempt spectral initialisation from the symmetrised graph.
    ///
    /// Uses the bottom non-trivial eigenvectors of the normalised graph
    /// Laplacian.  Returns true on success, false if the decomposition
    /// fails or the graph is degenerate.
    bool try_spectral_init(const MatrixType& P,
                           Eigen::Index n,
                           std::mt19937_64& /*rng*/) {
        if (n < n_components_ + 1) return false;

        // Degree and normalised Laplacian
        VectorType degree = P.rowwise().sum();
        VectorType d_inv_sqrt(n);
        for (Eigen::Index i = 0; i < n; ++i) {
            d_inv_sqrt(i) = (degree(i) > Scalar{0})
                                ? Scalar{1} / std::sqrt(degree(i))
                                : Scalar{0};
        }

        MatrixType L_norm = MatrixType::Identity(n, n);
        for (Eigen::Index i = 0; i < n; ++i) {
            for (Eigen::Index j = 0; j < n; ++j) {
                L_norm(i, j) -= d_inv_sqrt(i) * P(i, j) * d_inv_sqrt(j);
            }
        }

        Eigen::SelfAdjointEigenSolver<MatrixType> eig(L_norm);
        if (eig.info() != Eigen::Success) return false;

        // Use eigenvectors 1..n_components (skip the constant one)
        int nc = std::min(n_components_, static_cast<int>(n - 1));
        embedding_ = eig.eigenvectors().middleCols(1, nc);

        // Small scale to avoid large initial gradients
        Scalar scale = Scalar{1e-4} /
                       std::max(embedding_.cwiseAbs().maxCoeff(), Scalar{1e-10});
        embedding_ *= scale;

        return true;
    }
};

/// @}

} // namespace Skigen

#endif // SKIGEN_MANIFOLD_UMAP_H
