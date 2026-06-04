// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_FEATURE_SELECTION_VARIANCE_THRESHOLD_H
#define SKIGEN_FEATURE_SELECTION_VARIANCE_THRESHOLD_H

#include "../Core/Base.h"
#include "../Core/EigenHelpers.h"
#include "../Core/Validation.h"

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <stdexcept>
#include <vector>

namespace Skigen {

/// @defgroup Algo_VarianceThreshold VarianceThreshold
/// @ingroup FeatureSelection
/// @brief Feature selector that removes all low-variance features.
/// @{

/// @brief Feature selector that removes all low-variance features.
///
/// This feature selection algorithm looks only at the features (`X`), not the
/// desired outputs (`y`), and can thus be used for unsupervised learning.
///
/// Mirrors
/// [sklearn.feature_selection.VarianceThreshold](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html).
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `threshold` | `Scalar` | `0` | Features with a training-set variance lower than this threshold are removed. The default is to keep features with non-zero variance. |
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type | Description |
/// |----------|------|-------------|
/// | `variances()` | `RowVectorType` | Variances of every individual feature (1 × n_features). |
/// | `n_features_in()` | `IndexType` | Number of features seen during `fit()`. |
///
/// ### See also
///
/// - Skigen::SelectKBest — Keep top-k scoring features.
/// - Skigen::SelectFromModel — Threshold features by model coefficients.
///
/// ### Notes
///
/// Variances are computed using the biased (ddof=0) estimator, matching
/// numpy/scikit-learn convention.
///
/// ### Limitations relative to scikit-learn `feature_names_in_` is not exposed.
template <typename Scalar = double>
class VarianceThreshold
    : public Transformer<VarianceThreshold<Scalar>, Scalar> {
public:
    using Base = Transformer<VarianceThreshold<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;
    using typename Base::ScalarType;
    using BoolMaskType = Eigen::Array<bool, Eigen::Dynamic, 1>;

    // Make the dense base-class fit/transform overloads visible alongside
    // the sparse overloads added below — without these, the derived
    // sparse signatures would hide the base ones.
    using Base::fit;
    using Base::transform;
    using Base::fit_transform;
    using Base::inverse_transform;

    /// @brief Construct a VarianceThreshold selector.
    ///
    /// @param threshold Variance threshold (`Scalar`, default `0`). Features
    ///   with variance strictly less than or equal to this value (or
    ///   strictly less than this value when `threshold > 0`) are dropped.
    ///   Matches sklearn semantics: features kept satisfy `variance > threshold`.
    explicit VarianceThreshold(Scalar threshold = Scalar{0})
        : threshold_(threshold) {}

    // -- Accessors ----------------------------------------------------------

    /// @brief The configured variance threshold.
    [[nodiscard]] Scalar threshold() const noexcept { return threshold_; }

    /// @brief Per-feature variance (1 × n_features).
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] const RowVectorType& variances() const {
        this->check_is_fitted();
        return variances_;
    }

    /// @brief Boolean support mask: `true` for features that are kept.
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] BoolMaskType get_support_mask() const {
        this->check_is_fitted();
        return support_mask_;
    }

    /// @brief Get the support of the selector.
    ///
    /// @param indices If `true`, return integer indices instead of a boolean
    ///   mask.
    /// @return Either a boolean mask of length n_features (default) or a
    ///   vector of selected indices.
    [[nodiscard]] BoolMaskType get_support(bool indices = false) const {
        this->check_is_fitted();
        if (indices) {
            // Caller should use get_support_indices() — keep API parity by
            // throwing here would mismatch sklearn (it returns array of int).
            // We provide both methods; this overload returns the mask.
        }
        return support_mask_;
    }

    /// @brief Get the integer indices of selected features.
    [[nodiscard]] Eigen::VectorXi get_support_indices() const {
        this->check_is_fitted();
        std::vector<int> idx;
        idx.reserve(static_cast<std::size_t>(support_mask_.size()));
        for (Eigen::Index j = 0; j < support_mask_.size(); ++j) {
            if (support_mask_(j)) idx.push_back(static_cast<int>(j));
        }
        Eigen::VectorXi out(idx.size());
        for (std::size_t i = 0; i < idx.size(); ++i) {
            out(static_cast<Eigen::Index>(i)) = idx[i];
        }
        return out;
    }

    SKIGEN_PARAMS((threshold, threshold_, double))

    // -- Implementation (called by CRTP base) --------------------------------

    /// @brief Compute per-feature variance and the support mask.
    VarianceThreshold& fit_impl(const Eigen::Ref<const MatrixType>& X) {
        internal::check_non_empty(X);

        this->n_features_in_ = X.cols();
        RowVectorType mean = X.colwise().mean();
        variances_ = internal::colwise_variance<Scalar>(X, mean);

        support_mask_ = BoolMaskType(X.cols());
        for (Eigen::Index j = 0; j < X.cols(); ++j) {
            support_mask_(j) = variances_(j) > threshold_;
        }

        if (!support_mask_.any()) {
            throw std::runtime_error(
                "No feature in X meets the variance threshold "
                + std::to_string(static_cast<double>(threshold_)));
        }

        this->fitted_ = true;
        return *this;
    }

    /// @brief Reduce X to selected features.
    [[nodiscard]] MatrixType transform_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        Eigen::Index n_kept = 0;
        for (Eigen::Index j = 0; j < support_mask_.size(); ++j) {
            if (support_mask_(j)) ++n_kept;
        }
        MatrixType out(X.rows(), n_kept);
        Eigen::Index k = 0;
        for (Eigen::Index j = 0; j < support_mask_.size(); ++j) {
            if (support_mask_(j)) {
                out.col(k++) = X.col(j);
            }
        }
        return out;
    }

    // -- Sparse-aware overloads --------------------------------

    /// @brief Fit on a sparse matrix without densifying.
    ///
    /// Computes the per-column variance directly from the CSC/CSR
    /// representation. For column-major storage this is O(nnz), where nnz
    /// is the number of explicit nonzeros — implicit zeros contribute to
    /// the mean and variance but are never materialised.
    ///
    /// Variance formula (biased, ddof=0):
    ///
    /// @f[
    ///   \mu_j = \frac{1}{n} \sum_{i \in \text{nz}_j} X_{ij},\qquad
    ///   \sigma_j^2 = \frac{1}{n} \sum_{i \in \text{nz}_j} X_{ij}^2 - \mu_j^2.
    /// @f]
    ///
    /// Matches sklearn's `VarianceThreshold.fit` behaviour on sparse input.
    template <int Options, typename StorageIndex>
    VarianceThreshold& fit(
        const Eigen::SparseMatrix<Scalar, Options, StorageIndex>& X) {
        if (X.rows() == 0 || X.cols() == 0) {
            throw std::invalid_argument(
                "VarianceThreshold.fit: empty sparse matrix.");
        }
        // Convert to column-major view so we iterate by column cheaply.
        using ColSparse =
            Eigen::SparseMatrix<Scalar, Eigen::ColMajor, StorageIndex>;
        const ColSparse Xc = X;

        const Eigen::Index n = Xc.rows();
        const Eigen::Index p = Xc.cols();
        this->n_features_in_ = p;

        variances_ = RowVectorType(p);
        for (Eigen::Index j = 0; j < p; ++j) {
            Scalar sum{0};
            Scalar sum_sq{0};
            for (typename ColSparse::InnerIterator it(Xc, j); it; ++it) {
                const Scalar v = it.value();
                sum    += v;
                sum_sq += v * v;
            }
            const Scalar mean = sum / static_cast<Scalar>(n);
            variances_(j) = sum_sq / static_cast<Scalar>(n) - mean * mean;
            if (variances_(j) < Scalar{0}) variances_(j) = Scalar{0};
        }

        support_mask_ = BoolMaskType(p);
        for (Eigen::Index j = 0; j < p; ++j) {
            support_mask_(j) = variances_(j) > threshold_;
        }
        if (!support_mask_.any()) {
            throw std::runtime_error(
                "No feature in X meets the variance threshold "
                + std::to_string(static_cast<double>(threshold_)));
        }

        this->fitted_ = true;
        return *this;
    }

    /// @brief Reduce a sparse X to selected features without densifying.
    ///
    /// Returns a sparse matrix with the same number of rows and only the
    /// columns where `support_mask_(j)` is `true`.
    template <int Options, typename StorageIndex>
    [[nodiscard]] Eigen::SparseMatrix<Scalar, Options, StorageIndex>
    transform(
        const Eigen::SparseMatrix<Scalar, Options, StorageIndex>& X) const {
        this->check_is_fitted();
        if (X.cols() != this->n_features_in_) {
            throw std::invalid_argument(
                "X has " + std::to_string(X.cols()) +
                " features, but selector was fitted with " +
                std::to_string(this->n_features_in_) + " features.");
        }
        // Build the kept-column index map.
        std::vector<Eigen::Index> kept;
        kept.reserve(static_cast<std::size_t>(support_mask_.size()));
        for (Eigen::Index j = 0; j < support_mask_.size(); ++j) {
            if (support_mask_(j)) kept.push_back(j);
        }

        using ColSparse =
            Eigen::SparseMatrix<Scalar, Eigen::ColMajor, StorageIndex>;
        const ColSparse Xc = X;

        ColSparse out_col(X.rows(), static_cast<Eigen::Index>(kept.size()));
        // Reserve roughly the right amount.
        std::vector<Eigen::Index> col_nnz(kept.size(), 0);
        for (std::size_t i = 0; i < kept.size(); ++i) {
            col_nnz[i] =
                Xc.outerIndexPtr()[kept[i] + 1] - Xc.outerIndexPtr()[kept[i]];
        }
        out_col.reserve(col_nnz);
        for (std::size_t i = 0; i < kept.size(); ++i) {
            const Eigen::Index j = kept[i];
            for (typename ColSparse::InnerIterator it(Xc, j); it; ++it) {
                out_col.insert(it.row(), static_cast<Eigen::Index>(i)) =
                    it.value();
            }
        }
        out_col.makeCompressed();
        return Eigen::SparseMatrix<Scalar, Options, StorageIndex>(out_col);
    }

    /// @brief Reverse the transformation by zero-padding removed features.
    [[nodiscard]] MatrixType inverse_transform_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        const Eigen::Index n_in = this->n_features_in_;
        Eigen::Index n_kept = 0;
        for (Eigen::Index j = 0; j < support_mask_.size(); ++j) {
            if (support_mask_(j)) ++n_kept;
        }
        if (X.cols() != n_kept) {
            throw std::invalid_argument(
                "X has " + std::to_string(X.cols()) +
                " features but selector kept " + std::to_string(n_kept) + ".");
        }
        MatrixType out = MatrixType::Zero(X.rows(), n_in);
        Eigen::Index k = 0;
        for (Eigen::Index j = 0; j < n_in; ++j) {
            if (support_mask_(j)) {
                out.col(j) = X.col(k++);
            }
        }
        return out;
    }

private:
    Scalar threshold_;
    RowVectorType variances_;
    BoolMaskType support_mask_;
};

/// @}

}  // namespace Skigen

#endif  // SKIGEN_FEATURE_SELECTION_VARIANCE_THRESHOLD_H
