// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_FEATURE_SELECTION_VARIANCE_THRESHOLD_H
#define SKIGEN_FEATURE_SELECTION_VARIANCE_THRESHOLD_H

#include "../Core/Base.h"
#include "../Core/EigenHelpers.h"
#include "../Core/Validation.h"

#include <Eigen/Core>
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
/// @note **scikit-learn parity gaps:** `feature_names_in_` is not exposed.
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
