// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

#ifndef SKIGEN_MANIFOLD_DETAIL_TRUNCATED_EIGENSOLVER_H
#define SKIGEN_MANIFOLD_DETAIL_TRUNCATED_EIGENSOLVER_H

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <stdexcept>
#include <string>

// Optional Spectra (header-only ARPACK-style truncated eigensolver) backend.
// Enabled only when SKIGEN_ENABLE_SPECTRA is defined AND the Spectra headers
// are reachable. Eigen remains the sole required dependency: with the flag
// off (the default), only the dense path below is compiled.
#if defined(SKIGEN_ENABLE_SPECTRA)
#if __has_include(<Spectra/SymEigsSolver.h>)
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/DenseSymMatProd.h>
#define SKIGEN_HAVE_SPECTRA 1
#endif
#endif

namespace Skigen {
namespace internal {

/// @brief sklearn-compatible eigen-solver selection.
///
/// Spelled to mirror scikit-learn's `eigen_solver` strings:
/// `"auto"`, `"arpack"`, `"dense"`.
enum class EigenSolver { Auto, Arpack, Dense };

/// @brief Parse an sklearn-style eigen-solver string into the enum.
/// @throws std::invalid_argument for an unknown name.
inline EigenSolver parse_eigen_solver(const std::string& name) {
    if (name == "auto") return EigenSolver::Auto;
    if (name == "arpack") return EigenSolver::Arpack;
    if (name == "dense") return EigenSolver::Dense;
    throw std::invalid_argument(
        "eigen_solver must be 'auto', 'arpack', or 'dense'; got '" + name +
        "'.");
}

/// @brief Whether the Spectra (ARPACK-style) truncated backend is compiled in.
inline constexpr bool spectra_available() {
#if defined(SKIGEN_HAVE_SPECTRA)
    return true;
#else
    return false;
#endif
}

/// @brief Compute the `k` algebraically-smallest eigenpairs of a symmetric
///   matrix `A`, ascending by eigenvalue.
///
/// With `solver == Dense` (or `Auto` when Spectra is unavailable, or when the
/// problem is small) this uses Eigen's dense `SelfAdjointEigenSolver` and
/// returns its bottom `k` eigenvectors — bit-for-bit the previous behaviour.
/// With `solver == Arpack` and the Spectra backend compiled in, it uses a
/// truncated iterative solve. The dense path is always the default.
///
/// @param A       Symmetric matrix (n × n).
/// @param k       Number of smallest eigenpairs to return.
/// @param solver  Backend selection (Auto / Arpack / Dense).
/// @param eigenvalues  [out] Ascending eigenvalues (length k).
/// @param eigenvectors [out] Corresponding eigenvectors (n × k).
template <typename Scalar>
void smallest_eigenpairs(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A, int k,
    EigenSolver solver,
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& eigenvalues,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& eigenvectors) {
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    const Eigen::Index n = A.rows();
    k = std::min(k, static_cast<int>(n));

#if defined(SKIGEN_HAVE_SPECTRA)
    // Use the truncated solver when arpack is requested, or when auto is
    // selected on a large problem (sklearn's heuristic prefers arpack when
    // n_components << n_samples).
    const bool use_arpack =
        (solver == EigenSolver::Arpack) ||
        (solver == EigenSolver::Auto && n > 200 && k < static_cast<int>(n) / 2);
    if (use_arpack) {
        // Spectra finds the smallest-algebraic eigenpairs directly.
        Spectra::DenseSymMatProd<Scalar> op(A);
        const int ncv = std::min(static_cast<int>(n),
                                 std::max(2 * k + 1, 20));
        Spectra::SymEigsSolver<Spectra::DenseSymMatProd<Scalar>> eigs(
            op, k, ncv);
        eigs.init();
        eigs.compute(Spectra::SortRule::SmallestAlge);
        if (eigs.info() == Spectra::CompInfo::Successful) {
            eigenvalues = eigs.eigenvalues();
            eigenvectors = eigs.eigenvectors();
            // Spectra returns descending; flip to ascending.
            eigenvalues = eigenvalues.reverse().eval();
            eigenvectors = eigenvectors.rowwise().reverse().eval();
            return;
        }
        // Fall through to the dense path on failure.
    }
#else
    (void)solver;
#endif

    // Dense fallback (default). SelfAdjointEigenSolver returns ascending
    // eigenvalues; take the bottom k.
    Eigen::SelfAdjointEigenSolver<Matrix> eig(A);
    if (eig.info() != Eigen::Success)
        throw std::runtime_error(
            "TruncatedEigensolver: dense eigendecomposition did not "
            "converge.");
    eigenvalues = eig.eigenvalues().head(k);
    eigenvectors = eig.eigenvectors().leftCols(k);
}

}  // namespace internal
}  // namespace Skigen

#endif  // SKIGEN_MANIFOLD_DETAIL_TRUNCATED_EIGENSOLVER_H
