// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_DECOMPOSITION_TRUNCATED_SVD_H
#define SKIGEN_DECOMPOSITION_TRUNCATED_SVD_H

#include "../Core/Base.h"
#include "../Core/Validation.h"

#include <Eigen/Core>
#include <Eigen/SVD>
#include <algorithm>

namespace Skigen {

/// TruncatedSVD — Dimensionality reduction using truncated SVD (no centering).
/// Unlike PCA, does not center data — suitable for sparse data.
/// Mirrors sklearn.decomposition.TruncatedSVD.
template <typename Scalar = double>
class TruncatedSVD
    : public Transformer<TruncatedSVD<Scalar>, Scalar> {
public:
    using Base = Transformer<TruncatedSVD<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;

    explicit TruncatedSVD(Eigen::Index n_components = 2)
        : n_components_(n_components) {}

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

    [[nodiscard]] MatrixType transform_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        return X * components_.transpose();
    }

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

} // namespace Skigen

#endif // SKIGEN_DECOMPOSITION_TRUNCATED_SVD_H
