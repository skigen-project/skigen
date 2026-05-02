// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_DECOMPOSITION_PCA_H
#define SKIGEN_DECOMPOSITION_PCA_H

#include "../Core/Base.h"
#include "../Core/Validation.h"

#include <Eigen/Core>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>

namespace Skigen {

/// @defgroup Algo_PCA PCA
/// @ingroup Decomposition
/// @brief Principal Component Analysis (PCA) via full SVD.
/// @{

/// @brief Principal component analysis (PCA).
///
/// Linear dimensionality reduction using Singular Value Decomposition
/// of the data to project it to a lower dimensional space. The input
/// data is centered but not scaled for each feature before applying
/// the SVD.
///
/// Mirrors
/// [sklearn.decomposition.PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `n_components` | `IndexType` | `0` | Number of components to keep. `0` means keep all. |
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type | Description |
/// |----------|------|-------------|
/// | `components()` | `MatrixType` | Principal axes in feature space (n_components × n_features). |
/// | `explained_variance()` | `VectorType` | Variance explained by each component. |
/// | `explained_variance_ratio()` | `VectorType` | Percentage of variance explained by each component. |
/// | `singular_values()` | `VectorType` | The singular values corresponding to each component. |
/// | `mean()` | `RowVectorType` | Per-feature empirical mean, estimated from the training set. |
/// | `n_components()` | `IndexType` | The actual number of components (may differ from requested if clamped). |
///
/// ### See also
///
/// - Skigen::TruncatedSVD — SVD without centering (suitable for sparse data).
///
/// @note **scikit-learn parity gaps:** The following sklearn constructor
///   parameters are not yet supported: `copy`, `whiten`, `svd_solver`
///   (only full SVD via JacobiSVD), `tol`, `iterated_power`, `n_oversamples`,
///   `power_iteration_normalizer`, `random_state`.
///   The following sklearn fitted attributes are not yet exposed:
///   `noise_variance_`, `n_samples_`, `n_features_in_`, `feature_names_in_`.
///
/// ### Examples
///
/// @snippet pca.cpp example_pca
template <typename Scalar = double>
class PCA
    : public Transformer<PCA<Scalar>, Scalar> {
public:
    using Base = Transformer<PCA<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;

    /// @brief Construct a PCA estimator.
    ///
    /// @param n_components Number of components to keep (`IndexType`, default `0`).
    ///   `0` means all components are kept.
    explicit PCA(Eigen::Index n_components = 0) : n_components_(n_components) {}

    // -- Accessors ----------------------------------------------------------

    /// @brief The actual number of components after fitting.
    [[nodiscard]] Eigen::Index n_components() const noexcept { return n_components_actual_; }

    /// @brief Principal axes in feature space (n_components × n_features).
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
    /// @brief Per-feature empirical mean (1 × n_features).
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] const RowVectorType& mean() const {
        this->check_is_fitted(); return mean_;
    }

    // -- Implementation (called by CRTP base) --------------------------------

    /// @brief Fit the model with X by computing the SVD.
    ///
    /// Centers the data and performs full SVD (JacobiSVD) to compute
    /// principal components, explained variance, and singular values.
    ///
    /// @param X Training data of shape (n_samples, n_features).
    /// @return Reference to the fitted transformer (`*this`).
    PCA& fit_impl(const Eigen::Ref<const MatrixType>& X) {
        internal::check_non_empty(X);

        this->n_features_in_ = X.cols();
        const Eigen::Index n = X.rows();
        const Eigen::Index p = X.cols();

        n_components_actual_ = (n_components_ > 0)
            ? std::min(n_components_, std::min(n, p))
            : std::min(n, p);

        // Center data
        mean_ = X.colwise().mean();
        MatrixType X_c = X.rowwise() - mean_;

        // Full SVD: X_c = U * S * V^T
        Eigen::JacobiSVD<MatrixType> svd(X_c, Eigen::ComputeThinU | Eigen::ComputeThinV);

        // Components = first k rows of V^T
        components_ = svd.matrixV().leftCols(n_components_actual_).transpose();

        // Singular values
        singular_values_ = svd.singularValues().head(n_components_actual_);

        // Explained variance = s² / (n - 1)
        const Scalar denom = static_cast<Scalar>(n - 1);
        explained_variance_ = (singular_values_.array().square() / denom).matrix();

        // Explained variance ratio
        Scalar total_var = (svd.singularValues().array().square() / denom).sum();
        if (total_var > Scalar{0}) {
            explained_variance_ratio_ = explained_variance_ / total_var;
        } else {
            explained_variance_ratio_ = VectorType::Zero(n_components_actual_);
        }

        this->fitted_ = true;
        return *this;
    }

    /// @brief Apply dimensionality reduction to X.
    ///
    /// Projects data onto the first `n_components` principal axes:
    /// @f$ Z = (X - \mu) V^\top @f$.
    ///
    /// @param X Data matrix of shape (n_samples, n_features).
    /// @return Transformed data of shape (n_samples, n_components).
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] MatrixType transform_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        // Project: Z = (X - mean) * components^T
        return (X.rowwise() - mean_) * components_.transpose();
    }

    /// @brief Transform data back to its original space.
    ///
    /// Approximately reconstructs: @f$ \hat{X} = Z V + \mu @f$.
    ///
    /// @param X Transformed data of shape (n_samples, n_components).
    /// @return Reconstructed data of shape (n_samples, n_features).
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] MatrixType inverse_transform_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        // Reconstruct: X_hat = Z * components + mean
        return (X * components_).rowwise() + mean_;
    }

private:
    Eigen::Index n_components_;
    Eigen::Index n_components_actual_ = 0;

    MatrixType components_;       // (n_components x n_features)
    VectorType explained_variance_;
    VectorType explained_variance_ratio_;
    VectorType singular_values_;
    RowVectorType mean_;
};

/// @}

} // namespace Skigen

#endif // SKIGEN_DECOMPOSITION_PCA_H
