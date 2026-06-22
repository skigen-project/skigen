// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_MANIFOLD_LOCALLY_LINEAR_EMBEDDING_H
#define SKIGEN_MANIFOLD_LOCALLY_LINEAR_EMBEDDING_H

#include "../Core/Base.h"
#include "../Core/Validation.h"
#include "Detail/TruncatedEigensolver.h"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace Skigen {

/// @addtogroup Algo_Manifold
/// @{

/// @brief Locally Linear Embedding (LLE).
///
/// Nonlinear dimensionality reduction that preserves local neighborhood
/// structure. Each point is reconstructed as a weighted linear combination
/// of its K nearest neighbors; the embedding is then found by minimising
/// the same reconstruction error in the low-dimensional space.
///
/// Mirrors
/// [sklearn.manifold.LocallyLinearEmbedding](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.LocallyLinearEmbedding.html).
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `n_components` | `int` | `2` | Number of dimensions in the embedding. |
/// | `n_neighbors` | `int` | `5` | Number of nearest neighbors per point. |
/// | `reg` | `Scalar` | `1e-3` | Regularisation for the local covariance solve. |
/// | `max_iter` | `int` | `100` | (Reserved) maximum iterations for eigen solver. |
/// | `tol` | `Scalar` | `1e-6` | (Reserved) convergence tolerance. |
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type | Description |
/// |----------|------|-------------|
/// | `embedding()` | `const MatrixType&` | Embedding vectors (n_samples x n_components). |
/// | `reconstruction_error()` | `Scalar` | Sum of squared reconstruction residuals. |
///
/// ### Algorithm — standard method
///
/// 1. Find K nearest neighbors for each point (brute-force Euclidean).
/// 2. Compute reconstruction weights W by solving the constrained
///    least-squares problem for each point so that the sum of weights
///    is one and only neighbors contribute.
/// 3. Construct the alignment matrix M = (I - W)^T (I - W).
/// 4. Eigendecompose M and take the bottom eigenvectors (excluding
///    the trivial zero-eigenvalue eigenvector) as the embedding.
///
/// ### Limitations relative to scikit-learn
///
/// The `standard`, `modified`, `hessian`, and `ltsa` LLE methods are
/// implemented. The following scikit-learn parameters are not supported:
/// `eigen_solver`, `hessian_tol`, `modified_tol`, `neighbors_algorithm`,
/// `n_jobs`.
///
/// ### Examples
///
/// @snippet locally_linear_embedding.cpp example_lle
template <typename Scalar = double>
class LocallyLinearEmbedding
    : public Transformer<LocallyLinearEmbedding<Scalar>, Scalar> {
public:
    using Base = Transformer<LocallyLinearEmbedding<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;

    /// @brief Construct a Locally Linear Embedding estimator.
    ///
    /// @param n_components Number of embedding dimensions (default `2`).
    /// @param n_neighbors Number of nearest neighbors (default `5`).
    /// @param reg Regularisation added to the local covariance matrix
    ///   (default `1e-3`).
    /// @param max_iter Reserved for iterative eigen solvers (default `100`).
    /// @param tol Convergence tolerance (default `1e-6`).
    /// @param method LLE variant, `"standard"` (default) or `"modified"`.
    /// @param eigen_solver Eigensolver backend: `"auto"`, `"arpack"`, or
    ///   `"dense"` (default `"auto"`). `"arpack"` is only active under
    ///   `SKIGEN_ENABLE_SPECTRA`; otherwise the dense solver is used.
    explicit LocallyLinearEmbedding(int n_components = 2,
                                   int n_neighbors = 5,
                                   Scalar reg = Scalar{1e-3},
                                   int max_iter = 100,
                                   Scalar tol = Scalar{1e-6},
                                   std::string method = "standard",
                                   std::string eigen_solver = "auto")
        : n_components_(n_components)
        , n_neighbors_(n_neighbors)
        , reg_(reg)
        , max_iter_(max_iter)
        , tol_(tol)
        , method_(std::move(method))
        , eigen_solver_(std::move(eigen_solver)) {}

    /// @brief The configured eigensolver name (`"auto"` / `"arpack"` / `"dense"`).
    [[nodiscard]] const std::string& eigen_solver() const noexcept {
        return eigen_solver_;
    }

    /// @brief LLE variant string (`"standard"` or `"modified"`).
    [[nodiscard]] const std::string& method() const noexcept { return method_; }

    // -- Accessors ----------------------------------------------------------

    /// @brief Embedding vectors of shape (n_samples, n_components).
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] const MatrixType& embedding() const {
        this->check_is_fitted();
        return embedding_;
    }

    /// @brief Sum of squared reconstruction residuals.
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] Scalar reconstruction_error() const {
        this->check_is_fitted();
        return reconstruction_error_;
    }

    SKIGEN_PARAMS(
        (n_components, n_components_, int),
        (n_neighbors,  n_neighbors_,  int),
        (reg,          reg_,          double),
        (max_iter,     max_iter_,     int),
        (tol,          tol_,          double),
        (method,       method_,       std::string),
        (eigen_solver, eigen_solver_, std::string))

    // -- Implementation (called by CRTP base) --------------------------------

    /// @brief Fit the LLE model: compute an embedding of X.
    ///
    /// @param X Input data of shape (n_samples, n_features).
    /// @return Reference to the fitted transformer (`*this`).
    LocallyLinearEmbedding& fit_impl(
        const Eigen::Ref<const MatrixType>& X) {
        internal::check_non_empty(X);

        this->n_features_in_ = X.cols();
        const Eigen::Index n = X.rows();
        const Eigen::Index K = static_cast<Eigen::Index>(n_neighbors_);

        if (K >= n) {
            throw std::invalid_argument(
                "n_neighbors (" + std::to_string(n_neighbors_) +
                ") must be less than n_samples (" +
                std::to_string(n) + ").");
        }
        if (n_components_ > n - 1) {
            throw std::invalid_argument(
                "n_components (" + std::to_string(n_components_) +
                ") must be less than n_samples (" +
                std::to_string(n) + ").");
        }
        if (method_ != "standard" && method_ != "modified" &&
            method_ != "hessian" && method_ != "ltsa") {
            throw std::invalid_argument(
                "LocallyLinearEmbedding: method must be 'standard', "
                "'modified', 'hessian', or 'ltsa'.");
        }
        const Eigen::Index d = static_cast<Eigen::Index>(n_components_);
        if ((method_ == "modified" || method_ == "ltsa") && K <= d) {
            throw std::invalid_argument(
                "LocallyLinearEmbedding: " + method_ +
                " LLE requires n_neighbors > n_components.");
        }
        if (method_ == "hessian") {
            const Eigen::Index hessian_min = 1 + d + d * (d + 1) / 2;
            if (K < hessian_min) {
                throw std::invalid_argument(
                    "LocallyLinearEmbedding: hessian LLE requires n_neighbors "
                    ">= 1 + n_components + n_components*(n_components+1)/2.");
            }
        }

        // --- Step 1: Pairwise distances and K nearest neighbors ---
        MatrixType D = pairwise_squared_distances(X);
        // neighbors(i, k) is the index of the k-th nearest neighbor of i.
        Eigen::Matrix<Eigen::Index, Eigen::Dynamic, Eigen::Dynamic>
            neighbors(n, K);
        find_k_nearest(D, neighbors, n, K);

        // --- Steps 2-3: build the alignment matrix M ---
        MatrixType M;
        if (method_ == "modified") {
            M = build_modified_alignment(X, neighbors, n, K);
        } else if (method_ == "hessian") {
            M = build_hessian_alignment(X, neighbors, n, K);
        } else if (method_ == "ltsa") {
            M = build_ltsa_alignment(X, neighbors, n, K);
        } else {
            M = build_standard_alignment(X, neighbors, n, K);
        }

        // Symmetrise M to guard against floating-point asymmetry.
        M = (M + M.transpose()) / Scalar{2};

        // Bottom eigenpairs of M: route through the shared helper (dense by
        // default; Spectra ARPACK-style only under SKIGEN_ENABLE_SPECTRA).
        // Ask for n_components + 1 and drop the trivial null eigenvector.
        const internal::EigenSolver solver =
            internal::parse_eigen_solver(eigen_solver_);
        VectorType evals;
        MatrixType evecs;
        internal::smallest_eigenpairs<Scalar>(
            M, n_components_ + 1, solver, evals, evecs);

        embedding_ = evecs.block(
            0, 1, n, static_cast<Eigen::Index>(n_components_));

        // Reconstruction error: sum of the n_components smallest
        // non-trivial eigenvalues.
        reconstruction_error_ = Scalar{0};
        for (int k = 1; k <= n_components_; ++k) {
            reconstruction_error_ += evals(k);
        }

        this->fitted_ = true;
        return *this;
    }

    /// @brief Return the stored embedding (same as fit result).
    ///
    /// Because standard LLE has no out-of-sample extension, `transform`
    /// returns the embedding computed during `fit`. X is expected to
    /// be the same data that was used for fitting.
    ///
    /// @param X Data matrix (must match the training data shape).
    /// @return The embedding of shape (n_samples, n_components).
    [[nodiscard]] MatrixType transform_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        (void)X;
        return embedding_;
    }

private:
    int n_components_;
    int n_neighbors_;
    Scalar reg_;
    int max_iter_;
    Scalar tol_;
    std::string method_;
    std::string eigen_solver_ = "auto";

    MatrixType embedding_;
    Scalar reconstruction_error_ = Scalar{0};

    /// @brief Build the standard LLE alignment matrix M = (I - W)^T (I - W).
    [[nodiscard]] MatrixType build_standard_alignment(
        const Eigen::Ref<const MatrixType>& X,
        const Eigen::Matrix<Eigen::Index, Eigen::Dynamic, Eigen::Dynamic>&
            neighbors,
        Eigen::Index n,
        Eigen::Index K) const {
        MatrixType W = MatrixType::Zero(n, n);
        compute_reconstruction_weights(X, neighbors, W, n, K);
        MatrixType IW = MatrixType::Identity(n, n) - W;
        return IW.transpose() * IW;
    }

    /// @brief Build the modified LLE alignment matrix.
    ///
    /// Modified LLE (Zhang & Wang, 2006) uses multiple local weight vectors
    /// taken from the near-null eigenvectors of each neighborhood Gram matrix
    /// and accumulates their outer products into the global alignment matrix.
    [[nodiscard]] MatrixType build_modified_alignment(
        const Eigen::Ref<const MatrixType>& X,
        const Eigen::Matrix<Eigen::Index, Eigen::Dynamic, Eigen::Dynamic>&
            neighbors,
        Eigen::Index n,
        Eigen::Index K) const {
        const Eigen::Index d = static_cast<Eigen::Index>(n_components_);
        const Eigen::Index extra = K - d;  // number of near-null directions
        MatrixType M = MatrixType::Zero(n, n);
        const VectorType ones = VectorType::Ones(K);

        for (Eigen::Index i = 0; i < n; ++i) {
            MatrixType Z(K, X.cols());
            for (Eigen::Index k = 0; k < K; ++k) {
                Z.row(k) = X.row(neighbors(i, k)) - X.row(i);
            }
            MatrixType C = Z * Z.transpose();
            const Scalar trace_C = C.trace();
            C += reg_ * (trace_C / static_cast<Scalar>(K)) *
                 MatrixType::Identity(K, K);

            Eigen::SelfAdjointEigenSolver<MatrixType> eig(C);
            // Smallest `extra` eigenvectors span the local null space.
            MatrixType V = eig.eigenvectors().leftCols(extra);

            // Orthogonalise each null vector against the constant vector and
            // accumulate the contribution H = V (I - 1 1^T / K).
            for (Eigen::Index c = 0; c < extra; ++c) {
                VectorType v = V.col(c);
                const Scalar s = v.sum();
                v -= (s / static_cast<Scalar>(K)) * ones;
                const Scalar norm = v.norm();
                if (norm > Scalar{1e-12}) v /= norm;
                for (Eigen::Index a = 0; a < K; ++a) {
                    for (Eigen::Index b = 0; b < K; ++b) {
                        M(neighbors(i, a), neighbors(i, b)) += v(a) * v(b);
                    }
                }
            }
        }
        return M;
    }

    /// @brief Build the LTSA (Local Tangent Space Alignment) matrix.
    ///
    /// For each neighborhood, the local tangent space is estimated by PCA on
    /// the centred neighbor coordinates; the alignment projector onto the
    /// complement of {constant vector ∪ tangent space} is accumulated into M.
    [[nodiscard]] MatrixType build_ltsa_alignment(
        const Eigen::Ref<const MatrixType>& X,
        const Eigen::Matrix<Eigen::Index, Eigen::Dynamic, Eigen::Dynamic>&
            neighbors,
        Eigen::Index n,
        Eigen::Index K) const {
        const Eigen::Index d = static_cast<Eigen::Index>(n_components_);
        MatrixType M = MatrixType::Zero(n, n);
        const MatrixType id_K = MatrixType::Identity(K, K);
        const Scalar inv_K = Scalar{1} / static_cast<Scalar>(K);

        for (Eigen::Index i = 0; i < n; ++i) {
            MatrixType Z(K, X.cols());
            for (Eigen::Index k = 0; k < K; ++k) {
                Z.row(k) = X.row(neighbors(i, k));
            }
            // Centre on the neighborhood centroid.
            const RowVectorType mean = Z.colwise().mean();
            Z.rowwise() -= mean;

            // Local tangent directions: top-d right singular vectors of Z.
            Eigen::JacobiSVD<MatrixType> svd(Z, Eigen::ComputeThinU);
            MatrixType U = svd.matrixU().leftCols(d);  // (K x d) tangent coords

            // Local alignment G = I - 1 1^T / K - U U^T.
            MatrixType G = id_K;
            G.array() -= inv_K;
            G.noalias() -= U * U.transpose();

            for (Eigen::Index a = 0; a < K; ++a) {
                for (Eigen::Index b = 0; b < K; ++b) {
                    M(neighbors(i, a), neighbors(i, b)) += G(a, b);
                }
            }
        }
        return M;
    }

    /// @brief Build the Hessian LLE (HLLE) alignment matrix.
    ///
    /// Each neighborhood contributes the null space of a local Hessian
    /// estimator built from a constant + linear + quadratic polynomial basis
    /// over the local tangent coordinates.
    [[nodiscard]] MatrixType build_hessian_alignment(
        const Eigen::Ref<const MatrixType>& X,
        const Eigen::Matrix<Eigen::Index, Eigen::Dynamic, Eigen::Dynamic>&
            neighbors,
        Eigen::Index n,
        Eigen::Index K) const {
        const Eigen::Index d = static_cast<Eigen::Index>(n_components_);
        const Eigen::Index dp = d * (d + 1) / 2;  // quadratic basis size
        const Eigen::Index basis = 1 + d + dp;
        MatrixType M = MatrixType::Zero(n, n);

        for (Eigen::Index i = 0; i < n; ++i) {
            MatrixType Z(K, X.cols());
            for (Eigen::Index k = 0; k < K; ++k) {
                Z.row(k) = X.row(neighbors(i, k));
            }
            const RowVectorType mean = Z.colwise().mean();
            Z.rowwise() -= mean;

            Eigen::JacobiSVD<MatrixType> svd(Z, Eigen::ComputeThinU);
            MatrixType U = svd.matrixU().leftCols(d);  // (K x d) tangent coords

            // Polynomial basis Phi (K x basis): [1 | U | quadratic cross-terms].
            MatrixType Phi(K, basis);
            Phi.col(0).setOnes();
            Phi.middleCols(1, d) = U;
            Eigen::Index col = 1 + d;
            for (Eigen::Index p = 0; p < d; ++p) {
                for (Eigen::Index q = p; q < d; ++q) {
                    Phi.col(col++) =
                        U.col(p).cwiseProduct(U.col(q));
                }
            }

            // Orthonormalise Phi (Gram-Schmidt via thin QR); the last dp
            // columns form the local Hessian estimator H.
            Eigen::HouseholderQR<MatrixType> qr(Phi);
            MatrixType Q = qr.householderQ() * MatrixType::Identity(K, basis);
            MatrixType H = Q.rightCols(dp);

            MatrixType local = H * H.transpose();
            for (Eigen::Index a = 0; a < K; ++a) {
                for (Eigen::Index b = 0; b < K; ++b) {
                    M(neighbors(i, a), neighbors(i, b)) += local(a, b);
                }
            }
        }
        return M;
    }

    // -- Helpers ------------------------------------------------------------

    /// @brief Compute the n x n matrix of pairwise squared Euclidean distances.
    static MatrixType pairwise_squared_distances(
        const Eigen::Ref<const MatrixType>& X) {
        const Eigen::Index n = X.rows();
        VectorType sq_norms = X.rowwise().squaredNorm();
        MatrixType D = sq_norms.replicate(1, n)
                       + sq_norms.transpose().replicate(n, 1)
                       - Scalar{2} * (X * X.transpose());
        D = D.cwiseMax(Scalar{0});
        return D;
    }

    /// @brief Find K nearest neighbors for each point using brute-force
    ///   selection on the precomputed distance matrix.
    static void find_k_nearest(
        const MatrixType& D,
        Eigen::Matrix<Eigen::Index, Eigen::Dynamic, Eigen::Dynamic>& neighbors,
        Eigen::Index n, Eigen::Index K) {
        // For each row, partial-sort indices by distance.
        std::vector<Eigen::Index> indices(static_cast<std::size_t>(n));
        for (Eigen::Index i = 0; i < n; ++i) {
            std::iota(indices.begin(), indices.end(), Eigen::Index{0});
            // Put the K+1 smallest-distance indices in front (includes self).
            std::nth_element(
                indices.begin(),
                indices.begin() + K + 1,
                indices.end(),
                [&](Eigen::Index a, Eigen::Index b) {
                    return D(i, a) < D(i, b);
                });
            // Sort the first K+1 so self (distance 0) comes first.
            std::sort(indices.begin(), indices.begin() + K + 1,
                      [&](Eigen::Index a, Eigen::Index b) {
                          return D(i, a) < D(i, b);
                      });
            // Fill neighbors, skipping self.
            Eigen::Index count = 0;
            for (Eigen::Index j = 0; j <= K && count < K; ++j) {
                if (indices[static_cast<std::size_t>(j)] != i) {
                    neighbors(i, count++) =
                        indices[static_cast<std::size_t>(j)];
                }
            }
        }
    }

    /// @brief Compute reconstruction weights for each point from its
    ///   K nearest neighbors.
    ///
    /// Solves the constrained least-squares problem
    ///   min ||x_i - sum_j w_j x_{nb_j}||^2  s.t. sum_j w_j = 1
    /// via the local covariance matrix with Tikhonov regularisation.
    void compute_reconstruction_weights(
        const Eigen::Ref<const MatrixType>& X,
        const Eigen::Matrix<Eigen::Index, Eigen::Dynamic, Eigen::Dynamic>&
            neighbors,
        MatrixType& W,
        Eigen::Index n,
        Eigen::Index K) const {
        VectorType ones = VectorType::Ones(K);

        for (Eigen::Index i = 0; i < n; ++i) {
            // Z(k, d) = x_{nb_k} - x_i for each neighbor k.
            MatrixType Z(K, X.cols());
            for (Eigen::Index k = 0; k < K; ++k) {
                Z.row(k) = X.row(neighbors(i, k)) - X.row(i);
            }

            // Local covariance C = Z Z^T (K x K).
            MatrixType C = Z * Z.transpose();

            // Regularise: C += reg * trace(C) / K * I.
            Scalar trace_C = C.trace();
            C += reg_ * (trace_C / static_cast<Scalar>(K)) *
                 MatrixType::Identity(K, K);

            // Solve C w = 1, then normalise so sum(w) = 1.
            VectorType w = C.ldlt().solve(ones);
            Scalar w_sum = w.sum();
            if (std::abs(w_sum) < Scalar{1e-30}) {
                w.setConstant(Scalar{1} / static_cast<Scalar>(K));
            } else {
                w /= w_sum;
            }

            for (Eigen::Index k = 0; k < K; ++k) {
                W(i, neighbors(i, k)) = w(k);
            }
        }
    }
};

/// @}

} // namespace Skigen

#endif // SKIGEN_MANIFOLD_LOCALLY_LINEAR_EMBEDDING_H
