// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_DECOMPOSITION_PCA_H
#define SKIGEN_DECOMPOSITION_PCA_H

#include "../Core/Base.h"
#include "../Core/Validation.h"
#include "Detail/RandomizedSVD.h"

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/SparseCore>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>

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
/// | `svd_solver` | `std::string` | `"full"` | `"full"` (JacobiSVD) or `"randomized"` (Halko-Martinsson-Tropp). |
///
/// ### Sparse input
///
/// Sparse matrices are supported natively via implicit centering: the data
/// is mean-centered through a linear operator, so the input is never
/// materialised dense. Sparse fitting always uses the randomized solver
/// (mirroring sklearn, where centering a sparse matrix densifies it unless
/// the implicitly-centered randomized path is used).
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
/// ### Limitations relative to scikit-learn
///
/// The following scikit-learn constructor
///   parameters are not honoured: `copy`, `whiten`, `tol`,
///   `power_iteration_normalizer`. Only `svd_solver` values `"full"`
///   and `"randomized"` are supported (`"arpack"` / `"covariance_eigh"`
///   are not). The following sklearn fitted attributes are not exposed:
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
    using Base::fit;

    /// @brief Construct a PCA estimator.
    ///
    /// @param n_components Number of components to keep (`IndexType`, default `0`).
    ///   `0` means all components are kept.
    /// @param svd_solver Solver: `"full"` (default, exact JacobiSVD) or
    ///   `"randomized"` (Halko-Martinsson-Tropp). Sparse input always uses
    ///   the randomized path regardless of this setting.
    /// @param n_oversamples Extra random dimensions for the randomized
    ///   solver (default `10`, sklearn parity).
    /// @param n_iter Power iterations for the randomized solver
    ///   (default `5`, sklearn's "randomized" default for small data).
    /// @param random_state Optional seed for the randomized solver.
    explicit PCA(Eigen::Index n_components = 0,
                 std::string svd_solver = "full",
                 int n_oversamples = 10,
                 int n_iter = 5,
                 std::optional<uint64_t> random_state = std::nullopt)
        : n_components_(n_components),
          svd_solver_(std::move(svd_solver)),
          n_oversamples_(n_oversamples),
          n_iter_(n_iter),
          random_state_(random_state) {}

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

    /// @brief The configured SVD solver (`"full"` or `"randomized"`).
    [[nodiscard]] const std::string& svd_solver() const noexcept {
        return svd_solver_;
    }

    SKIGEN_PARAMS((n_components, n_components_, int),
                  (svd_solver, svd_solver_, std::string),
                  (n_oversamples, n_oversamples_, int),
                  (n_iter, n_iter_, int))

    // -- Implementation (called by CRTP base) --------------------------------

    /// @brief Fit the model with dense X.
    ///
    /// Centers the data, then computes the SVD using the configured solver:
    /// `"full"` (exact JacobiSVD) or `"randomized"` (Halko-Martinsson-Tropp).
    ///
    /// @param X Training data of shape (n_samples, n_features).
    /// @return Reference to the fitted transformer (`*this`).
    /// @throws std::invalid_argument for an unknown `svd_solver`.
    PCA& fit_impl(const Eigen::Ref<const MatrixType>& X) {
        internal::check_non_empty(X);

        this->n_features_in_ = X.cols();
        const Eigen::Index n = X.rows();
        const Eigen::Index p = X.cols();

        n_components_actual_ = (n_components_ > 0)
            ? std::min(n_components_, std::min(n, p))
            : std::min(n, p);

        // Center data.
        mean_ = X.colwise().mean();
        MatrixType X_c = X.rowwise() - mean_;

        const Scalar denom = static_cast<Scalar>(n - 1);

        if (svd_solver_ == "full" || svd_solver_ == "auto") {
            // Exact full SVD: X_c = U S V^T. Total variance from all
            // singular values for an exact explained-variance ratio.
            Eigen::JacobiSVD<MatrixType> svd(
                X_c, Eigen::ComputeThinU | Eigen::ComputeThinV);
            components_ =
                svd.matrixV().leftCols(n_components_actual_).transpose();
            singular_values_ = svd.singularValues().head(n_components_actual_);
            explained_variance_ =
                (singular_values_.array().square() / denom).matrix();
            const Scalar total_var =
                (svd.singularValues().array().square() / denom).sum();
            explained_variance_ratio_ = (total_var > Scalar{0})
                ? (explained_variance_ / total_var).eval()
                : VectorType::Zero(n_components_actual_);
        } else if (svd_solver_ == "randomized") {
            std::mt19937_64 rng(random_state_.value_or(
                static_cast<uint64_t>(std::random_device{}())));
            internal::DenseLinearOperator<Scalar> op(X_c);
            const auto svd = internal::randomized_svd<Scalar>(
                op, n_components_actual_, n_oversamples_, n_iter_, rng);
            components_ = svd.V.transpose();
            singular_values_ = svd.singular_values;
            explained_variance_ =
                (singular_values_.array().square() / denom).matrix();
            // Total variance is exact here: the full centered Gram trace.
            const Scalar total_var =
                X_c.array().square().sum() / denom;
            explained_variance_ratio_ = (total_var > Scalar{0})
                ? (explained_variance_ / total_var).eval()
                : VectorType::Zero(n_components_actual_);
        } else {
            throw std::invalid_argument(
                "PCA.fit: unknown svd_solver '" + svd_solver_ +
                "'. Supported: 'full', 'randomized'.");
        }

        this->fitted_ = true;
        return *this;
    }

    /// @brief Fit natively from a sparse design matrix without densifying.
    ///
    /// Computes the per-feature mean from the sparse matrix, then runs the
    /// randomized SVD against an implicitly-centered linear operator
    /// (@f$X - \mathbf{1}\mu@f$). The sparse input is never materialised
    /// dense. Mirrors sklearn's sparse PCA randomized path.
    template <int Options, typename StorageIndex>
    PCA& fit(const Eigen::SparseMatrix<Scalar, Options, StorageIndex>& X) {
        if (X.rows() == 0 || X.cols() == 0)
            throw std::invalid_argument("PCA.fit: empty sparse matrix.");

        this->n_features_in_ = X.cols();
        const Eigen::Index n = X.rows();
        const Eigen::Index p = X.cols();
        n_components_actual_ = (n_components_ > 0)
            ? std::min(n_components_, std::min(n, p))
            : std::min(n, p);

        // Column-major sparse view for efficient X^T y / X y.
        using ColSparse =
            Eigen::SparseMatrix<Scalar, Eigen::ColMajor, StorageIndex>;
        const ColSparse Xc = X;

        // Per-feature mean without densifying: column sums / n.
        mean_ = RowVectorType::Zero(p);
        for (Eigen::Index j = 0; j < Xc.outerSize(); ++j) {
            Scalar s{0};
            for (typename ColSparse::InnerIterator it(Xc, j); it; ++it)
                s += it.value();
            mean_(j) = s / static_cast<Scalar>(n);
        }

        std::mt19937_64 rng(random_state_.value_or(
            static_cast<uint64_t>(std::random_device{}())));
        internal::CenteredSparseLinearOperator<Scalar, Eigen::ColMajor,
                                               StorageIndex>
            op(Xc, mean_);
        const auto svd = internal::randomized_svd<Scalar>(
            op, n_components_actual_, n_oversamples_, n_iter_, rng);

        components_ = svd.V.transpose();
        singular_values_ = svd.singular_values;

        const Scalar denom = static_cast<Scalar>(n - 1);
        explained_variance_ =
            (singular_values_.array().square() / denom).matrix();

        // Exact total centered variance = sum_j (sum_i x_ij^2 - n mu_j^2).
        Scalar total_var{0};
        for (Eigen::Index j = 0; j < Xc.outerSize(); ++j) {
            Scalar sq{0};
            for (typename ColSparse::InnerIterator it(Xc, j); it; ++it)
                sq += it.value() * it.value();
            total_var += sq - static_cast<Scalar>(n) * mean_(j) * mean_(j);
        }
        total_var /= denom;
        explained_variance_ratio_ = (total_var > Scalar{0})
            ? (explained_variance_ / total_var).eval()
            : VectorType::Zero(n_components_actual_);

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
    std::string svd_solver_ = "full";
    int n_oversamples_ = 10;
    int n_iter_ = 5;
    std::optional<uint64_t> random_state_;
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
