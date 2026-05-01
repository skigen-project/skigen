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

/// PCA — Principal Component Analysis via full SVD.
/// Mirrors sklearn.decomposition.PCA.
template <typename Scalar = double>
class PCA
    : public Transformer<PCA<Scalar>, Scalar> {
public:
    using Base = Transformer<PCA<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;

    /// @param n_components Number of components to keep. 0 = keep all.
    explicit PCA(Eigen::Index n_components = 0) : n_components_(n_components) {}

    // -- Accessors ----------------------------------------------------------

    [[nodiscard]] Eigen::Index n_components() const noexcept { return n_components_actual_; }

    [[nodiscard]] const MatrixType& components() const {
        this->check_is_fitted(); return components_;
    }
    [[nodiscard]] const VectorType& explained_variance() const {
        this->check_is_fitted(); return explained_variance_;
    }
    [[nodiscard]] const VectorType& explained_variance_ratio() const {
        this->check_is_fitted(); return explained_variance_ratio_;
    }
    [[nodiscard]] const VectorType& singular_values() const {
        this->check_is_fitted(); return singular_values_;
    }
    [[nodiscard]] const RowVectorType& mean() const {
        this->check_is_fitted(); return mean_;
    }

    // -- Implementation (called by CRTP base) --------------------------------

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

    [[nodiscard]] MatrixType transform_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        // Project: Z = (X - mean) * components^T
        return (X.rowwise() - mean_) * components_.transpose();
    }

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

} // namespace Skigen

#endif // SKIGEN_DECOMPOSITION_PCA_H
