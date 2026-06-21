// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

#ifndef SKIGEN_DECOMPOSITION_DETAIL_RANDOMIZED_SVD_H
#define SKIGEN_DECOMPOSITION_DETAIL_RANDOMIZED_SVD_H

#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <Eigen/SparseCore>

#include <algorithm>
#include <cstdint>
#include <random>

namespace Skigen {
namespace internal {

/// @brief Result of a randomized truncated SVD: A ≈ U S Vᵀ.
///
/// All factors are truncated to the requested rank. `U` is
/// (m × n_components), `S` is length n_components, `V` is
/// (n × n_components).
template <typename Scalar>
struct RandomizedSvdResult {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> U;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> singular_values;
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> V;
};

/// @brief Linear operator over a dense matrix @f$A = X@f$.
///
/// Exposes @f$A M@f$ and @f$A^\top M@f$ for the Halko-Martinsson-Tropp
/// randomized range finder. Holds a reference to `X`; the caller must keep
/// it alive for the operator's lifetime.
template <typename Scalar>
class DenseLinearOperator {
public:
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    explicit DenseLinearOperator(const Eigen::Ref<const Matrix>& X) : X_(X) {}

    [[nodiscard]] Eigen::Index rows() const noexcept { return X_.rows(); }
    [[nodiscard]] Eigen::Index cols() const noexcept { return X_.cols(); }

    /// @brief Return @f$X M@f$ (M is cols(A) × k).
    [[nodiscard]] Matrix apply(const Matrix& M) const { return X_ * M; }
    /// @brief Return @f$X^\top M@f$ (M is rows(A) × k).
    [[nodiscard]] Matrix apply_transpose(const Matrix& M) const {
        return X_.transpose() * M;
    }

private:
    Eigen::Ref<const Matrix> X_;
};

/// @brief Plain sparse linear operator @f$A = X@f$ (no centering).
///
/// Exposes @f$X M@f$ and @f$X^\top M@f$ directly on the compressed sparse
/// representation. Holds a reference to `X`; the caller must keep it alive.
template <typename Scalar, int Options, typename StorageIndex>
class SparseLinearOperator {
public:
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using SparseType = Eigen::SparseMatrix<Scalar, Options, StorageIndex>;

    explicit SparseLinearOperator(const SparseType& X) : X_(X) {}

    [[nodiscard]] Eigen::Index rows() const noexcept { return X_.rows(); }
    [[nodiscard]] Eigen::Index cols() const noexcept { return X_.cols(); }

    [[nodiscard]] Matrix apply(const Matrix& M) const { return X_ * M; }
    [[nodiscard]] Matrix apply_transpose(const Matrix& M) const {
        return X_.transpose() * M;
    }

private:
    const SparseType& X_;
};

/// @brief Implicitly-centered sparse linear operator @f$A = X - 1\,\mu@f$.
///
/// Computes products with the mean-centered design matrix without ever
/// materialising @f$X@f$ dense. Uses the identity
/// @f[
///   (X - \mathbf{1}\mu)\,M = X M - \mathbf{1}(\mu M),\qquad
///   (X - \mathbf{1}\mu)^\top M = X^\top M - \mu^\top(\mathbf{1}^\top M),
/// @f]
/// where @f$\mu@f$ is the (1 × n_features) column-mean row vector and
/// @f$\mathbf{1}@f$ is the all-ones column. Holds references to the sparse
/// matrix and the mean; the caller must keep both alive.
template <typename Scalar, int Options, typename StorageIndex>
class CenteredSparseLinearOperator {
public:
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using RowVector = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;
    using SparseType = Eigen::SparseMatrix<Scalar, Options, StorageIndex>;

    CenteredSparseLinearOperator(const SparseType& X, const RowVector& mean)
        : X_(X), mean_(mean) {}

    [[nodiscard]] Eigen::Index rows() const noexcept { return X_.rows(); }
    [[nodiscard]] Eigen::Index cols() const noexcept { return X_.cols(); }

    /// @brief Return @f$(X - \mathbf{1}\mu)\,M@f$ (M is n_features × k).
    [[nodiscard]] Matrix apply(const Matrix& M) const {
        Matrix Y = X_ * M;                 // (m × k)
        const RowVector muM = mean_ * M;   // (1 × k)
        // Subtract the same row from every sample row.
        Y.rowwise() -= muM;
        return Y;
    }

    /// @brief Return @f$(X - \mathbf{1}\mu)^\top M@f$ (M is n_samples × k).
    [[nodiscard]] Matrix apply_transpose(const Matrix& M) const {
        Matrix Z = X_.transpose() * M;            // (n_features × k)
        const RowVector colsum = M.colwise().sum();  // (1 × k)
        // Z -= muᵀ · colsum  (outer product, n_features × k)
        Z.noalias() -= mean_.transpose() * colsum;
        return Z;
    }

private:
    const SparseType& X_;
    RowVector mean_;
};

/// @brief Halko-Martinsson-Tropp randomized truncated SVD.
///
/// Computes a rank-`n_components` approximation @f$A \approx U S V^\top@f$
/// for any operator exposing `rows()`, `cols()`, `apply(M)` (= @f$A M@f$),
/// and `apply_transpose(M)` (= @f$A^\top M@f$). The algorithm draws a
/// Gaussian test matrix of width `n_components + n_oversamples`, performs
/// `n_iter` QR-stabilised power iterations to capture the dominant
/// subspace, then runs a small dense SVD on the projected matrix.
///
/// Reference: Halko, Martinsson, Tropp, "Finding Structure with Randomness"
/// (SIAM Review, 2011), Algorithm 4.4 / 5.1.
///
/// @param A       Linear operator (dense or implicitly-centered sparse).
/// @param n_components  Target rank (clamped to min(rows, cols)).
/// @param n_oversamples Extra random dimensions for accuracy (sklearn: 10).
/// @param n_iter        Power iterations (sklearn: 5 for "randomized").
/// @param rng           Random engine for the Gaussian test matrix.
/// @return Truncated U, singular values, and V.
template <typename Scalar, typename Operator, typename Rng>
RandomizedSvdResult<Scalar> randomized_svd(const Operator& A,
                                           Eigen::Index n_components,
                                           int n_oversamples, int n_iter,
                                           Rng& rng) {
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    const Eigen::Index m = A.rows();
    const Eigen::Index n = A.cols();
    const Eigen::Index rank = std::min(n_components, std::min(m, n));
    const Eigen::Index k = std::min(
        rank + static_cast<Eigen::Index>(n_oversamples), std::min(m, n));

    // Gaussian test matrix Ω of shape (n × k).
    std::normal_distribution<Scalar> dist(Scalar{0}, Scalar{1});
    Matrix Omega(n, k);
    for (Eigen::Index j = 0; j < k; ++j)
        for (Eigen::Index i = 0; i < n; ++i) Omega(i, j) = dist(rng);

    // Y = A Ω, with QR-stabilised power iterations on (A Aᵀ)^q A Ω.
    Matrix Y = A.apply(Omega);  // (m × k)
    for (int q = 0; q < n_iter; ++q) {
        Eigen::HouseholderQR<Matrix> qr_y(Y);
        Matrix Q = qr_y.householderQ() * Matrix::Identity(m, k);
        Matrix Z = A.apply_transpose(Q);  // (n × k)
        Eigen::HouseholderQR<Matrix> qr_z(Z);
        Matrix Qz = qr_z.householderQ() * Matrix::Identity(n, k);
        Y = A.apply(Qz);  // (m × k)
    }

    // Orthonormal basis Q for the captured column space.
    Eigen::HouseholderQR<Matrix> qr_final(Y);
    Matrix Q = qr_final.householderQ() * Matrix::Identity(m, k);

    // B = Qᵀ A — small (k × n) matrix; SVD it densely.
    Matrix B = A.apply_transpose(Q).transpose();  // (k × n)

    Eigen::JacobiSVD<Matrix> svd(B, Eigen::ComputeThinU | Eigen::ComputeThinV);

    RandomizedSvdResult<Scalar> result;
    // U_A = Q · U_B, truncated to `rank` columns.
    result.U = (Q * svd.matrixU()).leftCols(rank);
    result.singular_values = svd.singularValues().head(rank);
    result.V = svd.matrixV().leftCols(rank);
    return result;
}

}  // namespace internal
}  // namespace Skigen

#endif  // SKIGEN_DECOMPOSITION_DETAIL_RANDOMIZED_SVD_H
