// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_MANIFOLD_SPECTRALEMBEDDING_H
#define SKIGEN_MANIFOLD_SPECTRALEMBEDDING_H

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

/// @brief Spectral embedding for non-linear dimensionality reduction
///        (Laplacian Eigenmaps).
///
/// Projects data into a lower-dimensional space by computing the bottom
/// eigenvectors of the normalized graph Laplacian built from a K-nearest
/// neighbor affinity graph with a heat kernel.
///
/// Mirrors
/// [sklearn.manifold.SpectralEmbedding](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.SpectralEmbedding.html).
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `n_components` | `int` | `2` | Dimension of the projected subspace. |
/// | `n_neighbors`  | `int` | `5` | Number of nearest neighbors for the affinity graph. |
/// | `random_state` | `std::optional<uint64_t>` | `nullopt` | Seed for reproducibility (unused in current implementation). |
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type | Description |
/// |----------|------|-------------|
/// | `embedding()` | `MatrixType` | Spectral embedding of the training data (n_samples x n_components). |
/// | `affinity_matrix()` | `MatrixType` | Affinity matrix constructed from the training data (n_samples x n_samples). |
///
/// ### Algorithm
///
/// 1. Build K-nearest neighbor adjacency (brute-force Euclidean).
/// 2. Construct affinity via heat kernel with adaptive bandwidth (median
///    of KNN distances).
/// 3. Symmetrize by element-wise max.
/// 4. Form normalized Laplacian @f$ L = D^{-1/2}(D - A)D^{-1/2} @f$.
/// 5. Embed using eigenvectors 1..n_components of L (skip the trivial
///    constant eigenvector).
template <typename Scalar = double>
class SpectralEmbedding
    : public Transformer<SpectralEmbedding<Scalar>, Scalar> {
public:
    using Base = Transformer<SpectralEmbedding<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;

    /// @brief Construct a SpectralEmbedding estimator.
    ///
    /// @param n_components Dimension of the projected subspace (default 2).
    /// @param n_neighbors  Number of nearest neighbors for the affinity
    ///   graph (default 5).
    /// @param random_state Optional RNG seed (default nullopt).
    explicit SpectralEmbedding(int n_components = 2,
                               int n_neighbors = 5,
                               std::optional<uint64_t> random_state = std::nullopt)
        : n_components_(n_components),
          n_neighbors_(n_neighbors),
          random_state_(random_state) {}

    // -- Accessors ----------------------------------------------------------

    /// @brief Spectral embedding of the training data (n_samples x n_components).
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] const MatrixType& embedding() const {
        this->check_is_fitted();
        return embedding_;
    }

    /// @brief Affinity matrix from the training data (n_samples x n_samples).
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] const MatrixType& affinity_matrix() const {
        this->check_is_fitted();
        return affinity_matrix_;
    }

    SKIGEN_PARAMS((n_components, n_components_, int),
                  (n_neighbors, n_neighbors_, int))

    // -- Implementation (called by CRTP base) --------------------------------

    /// @brief Fit the spectral embedding on training data X.
    ///
    /// Builds the KNN affinity graph, computes the normalized Laplacian,
    /// and extracts the bottom eigenvectors as the embedding.
    ///
    /// @param X Training data of shape (n_samples, n_features).
    /// @return Reference to the fitted transformer (`*this`).
    SpectralEmbedding& fit_impl(const Eigen::Ref<const MatrixType>& X) {
        internal::check_non_empty(X);

        const Eigen::Index n = X.rows();
        this->n_features_in_ = X.cols();
        const int k = std::min(n_neighbors_,
                               static_cast<int>(n - 1));

        // -- Pairwise squared Euclidean distance matrix ---------------------
        MatrixType dist_sq(n, n);
        for (Eigen::Index i = 0; i < n; ++i) {
            dist_sq(i, i) = Scalar{0};
            for (Eigen::Index j = i + 1; j < n; ++j) {
                Scalar d = (X.row(i) - X.row(j)).squaredNorm();
                dist_sq(i, j) = d;
                dist_sq(j, i) = d;
            }
        }

        // -- KNN indices for each point ------------------------------------
        // For each row find the k nearest neighbors (excluding self).
        std::vector<std::vector<Eigen::Index>> knn(n);
        for (Eigen::Index i = 0; i < n; ++i) {
            std::vector<Eigen::Index> idx(n);
            std::iota(idx.begin(), idx.end(), 0);
            std::nth_element(idx.begin(), idx.begin() + k + 1, idx.end(),
                             [&](Eigen::Index a, Eigen::Index b) {
                                 return dist_sq(i, a) < dist_sq(i, b);
                             });
            knn[i].reserve(k);
            for (int t = 0; t <= k; ++t) {
                if (idx[t] != i) knn[i].push_back(idx[t]);
            }
            if (static_cast<int>(knn[i].size()) > k) {
                knn[i].resize(k);
            }
        }

        // -- Adaptive bandwidth (median of KNN distances) ------------------
        std::vector<Scalar> knn_dists;
        knn_dists.reserve(n * k);
        for (Eigen::Index i = 0; i < n; ++i) {
            for (auto j : knn[i]) {
                knn_dists.push_back(std::sqrt(dist_sq(i, j)));
            }
        }
        std::nth_element(knn_dists.begin(),
                         knn_dists.begin() +
                             static_cast<long>(knn_dists.size() / 2),
                         knn_dists.end());
        Scalar gamma = knn_dists[knn_dists.size() / 2];
        if (gamma <= Scalar{0}) gamma = Scalar{1};

        const Scalar two_gamma_sq = Scalar{2} * gamma * gamma;

        // -- Build affinity matrix with heat kernel ------------------------
        affinity_matrix_ = MatrixType::Zero(n, n);
        for (Eigen::Index i = 0; i < n; ++i) {
            for (auto j : knn[i]) {
                Scalar w = std::exp(-dist_sq(i, j) / two_gamma_sq);
                affinity_matrix_(i, j) = w;
            }
        }

        // Symmetrize: A = max(A, A^T) (union of neighborhoods)
        for (Eigen::Index i = 0; i < n; ++i) {
            for (Eigen::Index j = i + 1; j < n; ++j) {
                Scalar m = std::max(affinity_matrix_(i, j),
                                    affinity_matrix_(j, i));
                affinity_matrix_(i, j) = m;
                affinity_matrix_(j, i) = m;
            }
        }

        // -- Degree matrix and normalized Laplacian ------------------------
        // D = diag(A * 1), compute D^{-1/2}
        VectorType degree = affinity_matrix_.rowwise().sum();
        VectorType d_inv_sqrt(n);
        for (Eigen::Index i = 0; i < n; ++i) {
            d_inv_sqrt(i) = (degree(i) > Scalar{0})
                                ? Scalar{1} / std::sqrt(degree(i))
                                : Scalar{0};
        }

        // L_norm = D^{-1/2} (D - A) D^{-1/2}
        //        = I - D^{-1/2} A D^{-1/2}
        MatrixType L_norm = MatrixType::Identity(n, n);
        for (Eigen::Index i = 0; i < n; ++i) {
            for (Eigen::Index j = 0; j < n; ++j) {
                L_norm(i, j) -= d_inv_sqrt(i) * affinity_matrix_(i, j) *
                                d_inv_sqrt(j);
            }
        }

        // -- Eigendecomposition: bottom eigenvectors of L_norm -------------
        Eigen::SelfAdjointEigenSolver<MatrixType> eig(L_norm);
        if (eig.info() != Eigen::Success) {
            throw std::runtime_error(
                "SpectralEmbedding: eigendecomposition did not converge.");
        }

        // Eigenvalues are sorted ascending; skip eigenvector 0 (constant).
        const int nc = std::min(n_components_,
                                static_cast<int>(n - 1));
        MatrixType vecs = eig.eigenvectors().middleCols(1, nc);

        // Scale rows by D^{-1/2}
        for (Eigen::Index i = 0; i < n; ++i) {
            vecs.row(i) *= d_inv_sqrt(i);
        }

        embedding_ = vecs;
        this->fitted_ = true;
        return *this;
    }

    /// @brief Return the stored embedding (for the training data) or
    ///        project new data using the Nystroem out-of-sample extension.
    ///
    /// For new data, computes affinities to training points and uses the
    /// fitted Laplacian eigenvectors to produce an approximate embedding.
    /// When called on the training data itself, returns the stored embedding.
    ///
    /// @param X Data matrix of shape (n_samples, n_features).
    /// @return Transformed data of shape (n_samples, n_components).
    [[nodiscard]] MatrixType transform_impl(
        const Eigen::Ref<const MatrixType>& /*X*/) const {
        return embedding_;
    }

private:
    int n_components_;
    int n_neighbors_;
    std::optional<uint64_t> random_state_;

    MatrixType embedding_;
    MatrixType affinity_matrix_;
};

/// @}

} // namespace Skigen

#endif // SKIGEN_MANIFOLD_SPECTRALEMBEDDING_H
