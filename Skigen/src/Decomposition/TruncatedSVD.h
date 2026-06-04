// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_DECOMPOSITION_TRUNCATED_SVD_H
#define SKIGEN_DECOMPOSITION_TRUNCATED_SVD_H

#include "../Core/Base.h"
#include "../Core/Validation.h"

#include <optional>

#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <Eigen/SparseCore>
#include <algorithm>
#include <random>

namespace Skigen {

/// @defgroup Algo_TruncatedSVD TruncatedSVD
/// @ingroup Decomposition
/// @brief Dimensionality reduction using truncated SVD (no centering).
/// @{

/// @brief Dimensionality reduction using truncated SVD (aka LSA).
///
/// This transformer performs linear dimensionality reduction by means
/// of truncated singular value decomposition (SVD). Contrary to PCA,
/// this estimator does **not** center the data before computing the
/// singular value decomposition. This means it can work with sparse
/// matrices efficiently.
///
/// Mirrors
/// [sklearn.decomposition.TruncatedSVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html).
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `n_components` | `IndexType` | `2` | Desired dimensionality of output data. |
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type | Description |
/// |----------|------|-------------|
/// | `components()` | `MatrixType` | The right singular vectors (n_components × n_features). |
/// | `explained_variance()` | `VectorType` | Variance explained by each component. |
/// | `explained_variance_ratio()` | `VectorType` | Percentage of variance explained. |
/// | `singular_values()` | `VectorType` | The singular values corresponding to each component. |
/// | `n_components()` | `IndexType` | The actual number of components. |
///
/// ### See also
///
/// - Skigen::PCA — PCA with centering (not suitable for sparse data).
///
/// ### Limitations relative to scikit-learn
///
/// The following scikit-learn constructor
///   parameters are not honoured: `algorithm` (only full SVD),
///   `n_iter`, `n_oversamples`, `power_iteration_normalizer`, `random_state`,
///   `tol`.
///   The following sklearn fitted attributes are not exposed:
///   `n_features_in_`, `feature_names_in_`.
///
/// ### Examples
///
/// @snippet truncated_svd.cpp example_truncated_svd
template <typename Scalar = double>
class TruncatedSVD
    : public Transformer<TruncatedSVD<Scalar>, Scalar> {
public:
    using Base = Transformer<TruncatedSVD<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;

    // Make the dense base-class fit/transform overloads visible alongside
    // the sparse overloads added below.
    using Base::fit;
    using Base::transform;
    using Base::fit_transform;
    using Base::inverse_transform;

    /// @brief Construct a TruncatedSVD estimator.
    ///
    /// @param n_components Desired dimensionality of output data
    ///   (`IndexType`, default `2`).
    explicit TruncatedSVD(Eigen::Index n_components = 2)
        : n_components_(n_components) {}

    /// @brief The actual number of components after fitting.
    [[nodiscard]] Eigen::Index n_components() const noexcept { return n_components_actual_; }

    /// @brief The right singular vectors (n_components × n_features).
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] const MatrixType& components() const {
        this->check_is_fitted(); return components_;
    }
    /// @brief Variance explained by each selected component.
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] const VectorType& explained_variance() const {
        this->check_is_fitted(); return explained_variance_;
    }
    /// @brief Percentage of variance explained by each selected component.
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] const VectorType& explained_variance_ratio() const {
        this->check_is_fitted(); return explained_variance_ratio_;
    }
    /// @brief Singular values corresponding to each component.
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] const VectorType& singular_values() const {
        this->check_is_fitted(); return singular_values_;
    }

    SKIGEN_PARAMS((n_components, n_components_, int))

    /// @brief Fit the model on training data X (no centering).
    ///
    /// @param X Training data of shape (n_samples, n_features).
    /// @return Reference to the fitted transformer (`*this`).
    TruncatedSVD& fit_impl(const Eigen::Ref<const MatrixType>& X) {
        internal::check_non_empty(X);
        this->n_features_in_ = X.cols();
        const Eigen::Index n = X.rows();
        const Eigen::Index p = X.cols();

        n_components_actual_ = std::min(n_components_, std::min(n, p));

        // SVD without centering
        Eigen::JacobiSVD<MatrixType> svd(X, Eigen::ComputeThinU | Eigen::ComputeThinV);

        components_ = svd.matrixV().leftCols(n_components_actual_).transpose();
        singular_values_ = svd.singularValues().head(n_components_actual_);

        const Scalar denom = static_cast<Scalar>(n - 1);
        explained_variance_ = (singular_values_.array().square() / denom).matrix();

        Scalar total_var = (svd.singularValues().array().square() / denom).sum();
        if (total_var > Scalar{0}) {
            explained_variance_ratio_ = explained_variance_ / total_var;
        } else {
            explained_variance_ratio_ = VectorType::Zero(n_components_actual_);
        }

        this->fitted_ = true;
        return *this;
    }

    // -- Sparse-aware overloads --------------------------------

    /// @brief Fit the model on a sparse matrix without densifying.
    ///
    /// Uses Halko-Martinsson-Tropp randomised SVD: draws a random Gaussian
    /// projection of width `n_components + n_oversamples`, runs `n_iter`
    /// power iterations with QR re-orthogonalisation for stability, then
    /// performs a small dense SVD on the projected matrix. All matrix
    /// products with `X` operate directly on the sparse representation
    /// — the input is never materialised as dense.
    ///
    /// Mirrors sklearn's `TruncatedSVD(algorithm="randomized")` with the
    /// defaults `n_oversamples=10`, `n_iter=5`,
    /// `power_iteration_normalizer="QR"`. Random state is consumed from
    /// `random_state` (defaults to a non-deterministic seed).
    template <int Options, typename StorageIndex>
    TruncatedSVD& fit(
        const Eigen::SparseMatrix<Scalar, Options, StorageIndex>& X,
        std::optional<uint64_t> random_state = std::nullopt,
        int n_oversamples = 10,
        int n_iter = 5) {
        if (X.rows() == 0 || X.cols() == 0) {
            throw std::invalid_argument(
                "TruncatedSVD.fit: empty sparse matrix.");
        }
        this->n_features_in_ = X.cols();
        const Eigen::Index n = X.rows();
        const Eigen::Index p = X.cols();
        n_components_actual_ = std::min(n_components_, std::min(n, p));

        // Convert to a column-major sparse view for efficient X^T y, X y.
        using ColSparse =
            Eigen::SparseMatrix<Scalar, Eigen::ColMajor, StorageIndex>;
        const ColSparse Xc = X;

        const Eigen::Index k = n_components_actual_ +
                               static_cast<Eigen::Index>(n_oversamples);
        const Eigen::Index k_eff = std::min(k, std::min(n, p));

        // Random Gaussian test matrix Ω of shape (p, k_eff).
        std::mt19937_64 rng(random_state.value_or(
            static_cast<uint64_t>(std::random_device{}())));
        std::normal_distribution<Scalar> dist(Scalar{0}, Scalar{1});

        MatrixType Omega(p, k_eff);
        for (Eigen::Index j = 0; j < k_eff; ++j) {
            for (Eigen::Index i = 0; i < p; ++i) {
                Omega(i, j) = dist(rng);
            }
        }

        // Y = X * Ω, with QR-stabilised power iterations.
        MatrixType Y = Xc * Omega;
        for (int q = 0; q < n_iter; ++q) {
            Eigen::HouseholderQR<MatrixType> qr_y(Y);
            MatrixType Q = qr_y.householderQ() * MatrixType::Identity(n, k_eff);
            // Z = X^T * Q ; Y = X * Z
            MatrixType Z = Xc.transpose() * Q;
            Eigen::HouseholderQR<MatrixType> qr_z(Z);
            MatrixType Qz = qr_z.householderQ() * MatrixType::Identity(p, k_eff);
            Y = Xc * Qz;
        }

        // Final orthonormal basis Q for the column space of Y.
        Eigen::HouseholderQR<MatrixType> qr_final(Y);
        MatrixType Q = qr_final.householderQ() * MatrixType::Identity(n, k_eff);

        // B = Q^T * X — small dense matrix of shape (k_eff, p).
        MatrixType B = Q.transpose() * Xc;

        Eigen::JacobiSVD<MatrixType> svd(
            B, Eigen::ComputeThinU | Eigen::ComputeThinV);

        // Truncate to n_components.
        components_ =
            svd.matrixV().leftCols(n_components_actual_).transpose();
        singular_values_ = svd.singularValues().head(n_components_actual_);

        const Scalar denom = static_cast<Scalar>(n - 1);
        explained_variance_ =
            (singular_values_.array().square() / denom).matrix();

        // Total variance: sum over all singular values of B (which captures
        // the captured-subspace energy). For sparse-randomised SVD we use
        // sum(B singular values²) as a proxy for the subspace's variance,
        // matching sklearn's randomised path.
        const Scalar total_var =
            (svd.singularValues().array().square() / denom).sum();
        if (total_var > Scalar{0}) {
            explained_variance_ratio_ = explained_variance_ / total_var;
        } else {
            explained_variance_ratio_ = VectorType::Zero(n_components_actual_);
        }

        this->fitted_ = true;
        return *this;
    }

    /// @brief Apply the fitted dimensionality reduction to a sparse matrix
    ///   without densifying. Returns a dense matrix of shape
    ///   (n_samples, n_components) (sklearn parity — the output is always
    ///   dense because it is low-rank).
    template <int Options, typename StorageIndex>
    [[nodiscard]] MatrixType transform(
        const Eigen::SparseMatrix<Scalar, Options, StorageIndex>& X) const {
        this->check_is_fitted();
        if (X.cols() != this->n_features_in_) {
            throw std::invalid_argument(
                "X has " + std::to_string(X.cols()) +
                " features, but TruncatedSVD was fitted with " +
                std::to_string(this->n_features_in_) + " features.");
        }
        return X * components_.transpose();
    }

    /// @brief Apply dimensionality reduction to X.
    ///
    /// @param X Data matrix of shape (n_samples, n_features).
    /// @return Transformed data of shape (n_samples, n_components).
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] MatrixType transform_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        return X * components_.transpose();
    }

    /// @brief Transform data back to its original space.
    ///
    /// @param X Transformed data of shape (n_samples, n_components).
    /// @return Reconstructed data of shape (n_samples, n_features).
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] MatrixType inverse_transform_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        return X * components_;
    }

private:
    Eigen::Index n_components_;
    Eigen::Index n_components_actual_ = 0;

    MatrixType components_;
    VectorType explained_variance_;
    VectorType explained_variance_ratio_;
    VectorType singular_values_;
};

/// @}

} // namespace Skigen

#endif // SKIGEN_DECOMPOSITION_TRUNCATED_SVD_H
