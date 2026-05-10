// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_CORE_SPARSE_SUPPORT_H
#define SKIGEN_CORE_SPARSE_SUPPORT_H

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <type_traits>

namespace Skigen {

/// @addtogroup Core
/// @{

namespace internal {

// ---------------------------------------------------------------------------
// Type traits — compile-time test for an Eigen sparse matrix.
// ---------------------------------------------------------------------------

template <typename T>
struct is_eigen_sparse : std::false_type {};

template <typename Scalar, int Options, typename StorageIndex>
struct is_eigen_sparse<Eigen::SparseMatrix<Scalar, Options, StorageIndex>>
    : std::true_type {};

template <typename T>
inline constexpr bool is_eigen_sparse_v = is_eigen_sparse<T>::value;

// ---------------------------------------------------------------------------
// Sparse column statistics
//
// These helpers extract the per-column quantities that recur in the
// implicit-centring sparse fits used by Ridge / Lasso / ElasticNet /
// LinearRegression / KMeans / VarianceThreshold / chi2 etc. They walk
// the CSC nonzeros once and never densify.
// ---------------------------------------------------------------------------

/// @brief Per-column sum of an Eigen::SparseMatrix.
///
/// Operates on a column-major view obtained by copy when necessary.
/// @return RowVector of length `X.cols()`.
template <typename Scalar, int Options, typename StorageIndex>
Eigen::Matrix<Scalar, 1, Eigen::Dynamic>
sparse_colwise_sum(
    const Eigen::SparseMatrix<Scalar, Options, StorageIndex>& X) {
    using ColSparse =
        Eigen::SparseMatrix<Scalar, Eigen::ColMajor, StorageIndex>;
    const ColSparse Xc = X;
    Eigen::Matrix<Scalar, 1, Eigen::Dynamic> out =
        Eigen::Matrix<Scalar, 1, Eigen::Dynamic>::Zero(Xc.cols());
    for (Eigen::Index j = 0; j < Xc.cols(); ++j) {
        Scalar s{0};
        for (typename ColSparse::InnerIterator it(Xc, j); it; ++it) {
            s += it.value();
        }
        out(j) = s;
    }
    return out;
}

/// @brief Per-column mean of an Eigen::SparseMatrix.
///
/// Implicit zeros contribute to the denominator (n_rows) but not the
/// numerator — the mean is `sum_nz / n_rows`.
template <typename Scalar, int Options, typename StorageIndex>
Eigen::Matrix<Scalar, 1, Eigen::Dynamic>
sparse_colwise_mean(
    const Eigen::SparseMatrix<Scalar, Options, StorageIndex>& X) {
    Eigen::Matrix<Scalar, 1, Eigen::Dynamic> s = sparse_colwise_sum(X);
    return s / static_cast<Scalar>(X.rows());
}

/// @brief Per-column squared L2 norm @f$ \|X[:, j]\|^2 @f$.
template <typename Scalar, int Options, typename StorageIndex>
Eigen::Matrix<Scalar, Eigen::Dynamic, 1>
sparse_colwise_squared_norm(
    const Eigen::SparseMatrix<Scalar, Options, StorageIndex>& X) {
    using ColSparse =
        Eigen::SparseMatrix<Scalar, Eigen::ColMajor, StorageIndex>;
    const ColSparse Xc = X;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> out =
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Zero(Xc.cols());
    for (Eigen::Index j = 0; j < Xc.cols(); ++j) {
        Scalar s{0};
        for (typename ColSparse::InnerIterator it(Xc, j); it; ++it) {
            s += it.value() * it.value();
        }
        out(j) = s;
    }
    return out;
}

/// @brief Per-column biased variance (`ddof = 0`).
///
/// @f$ \mathrm{Var}_j = \frac{1}{n}\!\sum_i X_{ij}^2 - \mu_j^2 @f$,
/// where @f$ \mu_j @f$ is the column mean computed by sparse iteration.
/// Tiny FP-cancellation negatives are clamped to zero.
template <typename Scalar, int Options, typename StorageIndex>
Eigen::Matrix<Scalar, 1, Eigen::Dynamic>
sparse_colwise_variance(
    const Eigen::SparseMatrix<Scalar, Options, StorageIndex>& X) {
    const Scalar n = static_cast<Scalar>(X.rows());
    Eigen::Matrix<Scalar, 1, Eigen::Dynamic> mean = sparse_colwise_mean(X);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> sqn  =
        sparse_colwise_squared_norm(X);
    Eigen::Matrix<Scalar, 1, Eigen::Dynamic> out(X.cols());
    for (Eigen::Index j = 0; j < X.cols(); ++j) {
        const Scalar v = sqn(j) / n - mean(j) * mean(j);
        out(j) = v < Scalar{0} ? Scalar{0} : v;
    }
    return out;
}

/// @brief Centred per-column squared norm:
///   @f$ \|X[:, j] - \bar{x}_j\|^2 = \|X[:, j]\|^2 - n\,\bar{x}_j^2 @f$.
///
/// This is the identity that the implicit-centring sparse linear models
/// (Ridge / LinearRegression / Lasso / ElasticNet) lean on so that the
/// centred Gram or per-feature squared norm can be recovered without
/// ever densifying X. Tiny FP-cancellation negatives are clamped to
/// zero.
template <typename Scalar, int Options, typename StorageIndex>
Eigen::Matrix<Scalar, Eigen::Dynamic, 1>
sparse_centered_squared_norm(
    const Eigen::SparseMatrix<Scalar, Options, StorageIndex>& X) {
    Eigen::Matrix<Scalar, 1, Eigen::Dynamic> mean = sparse_colwise_mean(X);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> sqn  =
        sparse_colwise_squared_norm(X);
    const Scalar n = static_cast<Scalar>(X.rows());
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> out(X.cols());
    for (Eigen::Index j = 0; j < X.cols(); ++j) {
        const Scalar v = sqn(j) - n * mean(j) * mean(j);
        out(j) = v < Scalar{0} ? Scalar{0} : v;
    }
    return out;
}

} // namespace internal

/// @}

} // namespace Skigen

#endif // SKIGEN_CORE_SPARSE_SUPPORT_H
