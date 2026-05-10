// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_PREPROCESSING_MAX_ABS_SCALER_H
#define SKIGEN_PREPROCESSING_MAX_ABS_SCALER_H

#include "../Core/Base.h"
#include "../Core/EigenHelpers.h"
#include "../Core/Validation.h"

#include <Eigen/Core>

namespace Skigen {

/// @defgroup Algo_MaxAbsScaler MaxAbsScaler
/// @ingroup Preprocessing
/// @brief Scale each feature by its maximum absolute value.
/// @{

/// @brief Scale each feature by its maximum absolute value.
///
/// This estimator scales and translates each feature individually such
/// that the maximal absolute value of each feature in the training set
/// will be 1.0. It does not shift/center the data, and thus does not
/// destroy any sparsity.
///
/// Mirrors
/// [sklearn.preprocessing.MaxAbsScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html).
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `clip` | `bool` | `false` | Set to `true` to clip transformed values to `[-1, 1]`. |
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type | Description |
/// |----------|------|-------------|
/// | `max_abs()` | `RowVectorType` | Per-feature maximum absolute value. |
/// | `scale()` | `RowVectorType` | Per-feature scaling factor (same as `max_abs()` but with zeros replaced by 1). |
/// | `n_samples_seen()` | `IndexType` | Number of samples processed by `fit()`. |
///
/// ### See also
///
/// - Skigen::StandardScaler — Standardize to zero mean and unit variance.
/// - Skigen::MinMaxScaler — Scale features to a given range.
///
/// ### Limitations relative to scikit-learn
///
/// The following scikit-learn constructor
///   parameters are not honoured: `copy`.
///   `partial_fit()` is not implemented.
///   The following sklearn fitted attributes are not exposed:
///   `n_features_in_`, `feature_names_in_`.
///
/// ### Examples
///
/// @snippet maxabs_scaler.cpp example_max_abs_scaler
template <typename Scalar = double>
class MaxAbsScaler
    : public Transformer<MaxAbsScaler<Scalar>, Scalar> {
public:
    using Base = Transformer<MaxAbsScaler<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;

    /// @brief Construct a MaxAbsScaler.
    ///
    /// @param clip Whether to clip transformed values to `[-1, 1]`
    ///   (`bool`, default `false`).
    explicit MaxAbsScaler(bool clip = false) : clip_(clip) {}

    // -- Accessors ----------------------------------------------------------

    /// @brief Whether clipping is enabled.
    [[nodiscard]] bool clip() const noexcept { return clip_; }

    /// @brief Per-feature maximum absolute value (1 × n_features).
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] const RowVectorType& max_abs() const {
        this->check_is_fitted(); return max_abs_;
    }
    /// @brief Per-feature scaling factor (1 × n_features).
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] const RowVectorType& scale() const {
        this->check_is_fitted(); return scale_;
    }
    /// @brief Number of samples processed during `fit()`.
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] IndexType n_samples_seen() const {
        this->check_is_fitted(); return n_samples_seen_;
    }

    // -- Implementation (called by CRTP base) --------------------------------

    /// @brief Compute per-feature maximum absolute value for later scaling.
    ///
    /// @param X Training data of shape (n_samples, n_features).
    /// @return Reference to the fitted transformer (`*this`).
    MaxAbsScaler& fit_impl(const Eigen::Ref<const MatrixType>& X) {
        internal::check_non_empty(X);

        this->n_features_in_ = X.cols();
        n_samples_seen_ = X.rows();

        max_abs_ = X.cwiseAbs().colwise().maxCoeff();
        scale_ = max_abs_;
        internal::handle_zeros_in_scale(scale_);

        this->fitted_ = true;
        return *this;
    }

    /// @brief Online update of `max_abs_` by extending the running per-feature
    ///   maximum-absolute-value with a new batch.
    ///
    /// Matches sklearn's `MaxAbsScaler.partial_fit` contract.
    ///
    /// @param X Batch of training data, shape (n_samples_batch, n_features).
    /// @return Reference to the fitted transformer (`*this`).
    /// @throws std::invalid_argument on feature-count mismatch or empty X.
    MaxAbsScaler& partial_fit(const Eigen::Ref<const MatrixType>& X) {
        internal::check_non_empty(X);

        if (!this->fitted_) {
            return fit_impl(X);
        }

        if (X.cols() != this->n_features_in_) {
            throw std::invalid_argument(
                "X has " + std::to_string(X.cols()) + " features, but "
                "partial_fit was previously called with " +
                std::to_string(this->n_features_in_) + " features.");
        }

        const RowVectorType batch_max_abs = X.cwiseAbs().colwise().maxCoeff();
        max_abs_ = max_abs_.cwiseMax(batch_max_abs);
        scale_ = max_abs_;
        internal::handle_zeros_in_scale(scale_);

        n_samples_seen_ += X.rows();
        return *this;
    }

    /// @brief Scale features of X by max absolute value.
    ///
    /// @param X Data matrix of shape (n_samples, n_features).
    /// @return Transformed data of same shape, scaled to `[-1, 1]`.
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] MatrixType transform_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        MatrixType result = X.array().rowwise() / scale_.array();
        if (clip_) {
            result = result.cwiseMax(Scalar{-1}).cwiseMin(Scalar{1});
        }
        return result;
    }

    /// @brief Scale back the data to the original representation.
    ///
    /// @param X Transformed data of shape (n_samples, n_features).
    /// @return Un-transformed data of same shape.
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] MatrixType inverse_transform_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        return X.array().rowwise() * scale_.array();
    }

    /// @brief Transform features in-place by scaling by max absolute value.
    /// @param X Data matrix of shape (n_samples, n_features), modified in place.
    /// @throws std::runtime_error if the model has not been fitted.
    void transform_inplace(Eigen::Ref<MatrixType> X) const {
        this->check_is_fitted();
        this->validate_feature_count(X);
        X.array().rowwise() /= scale_.array();
        if (clip_) {
            X = X.cwiseMax(Scalar{-1}).cwiseMin(Scalar{1});
        }
    }

    /// @brief Inverse-transform features in-place to original scale.
    /// @param X Scaled data matrix, modified in place.
    /// @throws std::runtime_error if the model has not been fitted.
    void inverse_transform_inplace(Eigen::Ref<MatrixType> X) const {
        this->check_is_fitted();
        this->validate_feature_count(X);
        X.array().rowwise() *= scale_.array();
    }

private:
    bool clip_;

    RowVectorType max_abs_;
    RowVectorType scale_;
    IndexType n_samples_seen_ = 0;
};

/// @}

} // namespace Skigen

#endif // SKIGEN_PREPROCESSING_MAX_ABS_SCALER_H
