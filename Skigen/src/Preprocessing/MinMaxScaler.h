// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_PREPROCESSING_MIN_MAX_SCALER_H
#define SKIGEN_PREPROCESSING_MIN_MAX_SCALER_H

#include "../Core/Base.h"
#include "../Core/EigenHelpers.h"
#include "../Core/Validation.h"

#include <Eigen/Core>
#include <stdexcept>
#include <utility>

namespace Skigen {

/// @defgroup Algo_MinMaxScaler MinMaxScaler
/// @ingroup Preprocessing
/// @brief Transform features by scaling each feature to a given range.
/// @{

/// @brief Transform features by scaling each feature to a given range.
///
/// This estimator scales and translates each feature individually such
/// that it is in the given range on the training set, e.g. between
/// zero and one (the default).
///
/// The transformation is:
///
/// @f[
///   X_{\text{scaled}} = \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}}
///     \cdot (\text{max} - \text{min}) + \text{min}
/// @f]
///
/// Mirrors
/// [sklearn.preprocessing.MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html).
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `feature_range` | `std::pair<Scalar,Scalar>` | `{0, 1}` | Desired range of transformed data. |
/// | `clip` | `bool` | `false` | Set to `true` to clip transformed values to the provided `feature_range`. |
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type | Description |
/// |----------|------|-------------|
/// | `data_min()` | `RowVectorType` | Per-feature minimum seen in the data. |
/// | `data_max()` | `RowVectorType` | Per-feature maximum seen in the data. |
/// | `data_range()` | `RowVectorType` | Per-feature range `data_max_ - data_min_`. |
/// | `scale()` | `RowVectorType` | Per-feature relative scaling. |
/// | `min()` | `RowVectorType` | Per-feature adjustment for minimum. |
/// | `n_samples_seen()` | `IndexType` | Number of samples processed by `fit()`. |
///
/// ### See also
///
/// - Skigen::StandardScaler — Standardize to zero mean and unit variance.
/// - Skigen::MaxAbsScaler — Scale by maximum absolute value.
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
/// @snippet minmax_scaler.cpp example_min_max_scaler
template <typename Scalar = double>
class MinMaxScaler
    : public Transformer<MinMaxScaler<Scalar>, Scalar> {
public:
    using Base = Transformer<MinMaxScaler<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;

    /// @brief Construct a MinMaxScaler.
    ///
    /// @param feature_range Desired range of transformed data
    ///   (`std::pair<Scalar,Scalar>`, default `{0, 1}`).
    /// @param clip Whether to clip transformed values to `feature_range`
    ///   (`bool`, default `false`).
    /// @throws std::invalid_argument if `feature_range.first >= feature_range.second`.
    explicit MinMaxScaler(std::pair<Scalar, Scalar> feature_range = {Scalar{0}, Scalar{1}},
                          bool clip = false)
        : feature_range_(feature_range), clip_(clip) {
        if (feature_range_.first >= feature_range_.second) {
            throw std::invalid_argument(
                "feature_range: min must be strictly less than max.");
        }
    }

    // -- Accessors ----------------------------------------------------------

    /// @brief Desired feature range.
    [[nodiscard]] const std::pair<Scalar, Scalar>& feature_range() const noexcept {
        return feature_range_;
    }
    /// @brief Whether clipping is enabled.
    [[nodiscard]] bool clip() const noexcept { return clip_; }

    /// @brief Per-feature minimum seen in the data.
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] const RowVectorType& data_min() const {
        this->check_is_fitted(); return data_min_;
    }
    /// @brief Per-feature maximum seen in the data.
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] const RowVectorType& data_max() const {
        this->check_is_fitted(); return data_max_;
    }
    /// @brief Per-feature range (`data_max_ - data_min_`).
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] const RowVectorType& data_range() const {
        this->check_is_fitted(); return data_range_;
    }
    /// @brief Per-feature relative scaling factor.
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] const RowVectorType& scale() const {
        this->check_is_fitted(); return scale_;
    }
    /// @brief Per-feature adjustment for minimum.
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] const RowVectorType& min() const {
        this->check_is_fitted(); return min_;
    }
    /// @brief Number of samples processed during `fit()`.
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] IndexType n_samples_seen() const {
        this->check_is_fitted(); return n_samples_seen_;
    }

    // -- Parameter reflection ---------------------------------------------------

    SKIGEN_PARAMS((clip, clip_, bool))

    // -- Implementation (called by CRTP base) --------------------------------

    /// @brief Compute per-feature min and max to be used for later scaling.
    ///
    /// @param X Training data of shape (n_samples, n_features).
    /// @return Reference to the fitted transformer (`*this`).
    MinMaxScaler& fit_impl(const Eigen::Ref<const MatrixType>& X) {
        internal::check_non_empty(X);

        this->n_features_in_ = X.cols();
        n_samples_seen_ = X.rows();

        data_min_ = X.colwise().minCoeff();
        data_max_ = X.colwise().maxCoeff();
        data_range_ = data_max_ - data_min_;

        // Handle zero-range features
        RowVectorType safe_range = data_range_;
        internal::handle_zeros_in_scale(safe_range);

        const Scalar range_min = feature_range_.first;
        const Scalar range_max = feature_range_.second;

        scale_ = (range_max - range_min) / safe_range.array();
        min_ = RowVectorType(range_min - (data_min_.array() * scale_.array()));

        this->fitted_ = true;
        return *this;
    }

    /// @brief Online update of `data_min_`, `data_max_`, and the scaling
    ///   parameters by extending the running per-feature extrema with a new
    ///   batch.
    ///
    /// Matches sklearn's `MinMaxScaler.partial_fit` contract: subsequent
    /// `partial_fit` calls produce the same fitted state as a single `fit`
    /// over the concatenated data.
    ///
    /// @param X Batch of training data, shape (n_samples_batch, n_features).
    /// @return Reference to the fitted transformer (`*this`).
    /// @throws std::invalid_argument on feature-count mismatch or empty X.
    MinMaxScaler& partial_fit(const Eigen::Ref<const MatrixType>& X) {
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

        const RowVectorType batch_min = X.colwise().minCoeff();
        const RowVectorType batch_max = X.colwise().maxCoeff();
        data_min_ = data_min_.cwiseMin(batch_min);
        data_max_ = data_max_.cwiseMax(batch_max);
        data_range_ = data_max_ - data_min_;

        RowVectorType safe_range = data_range_;
        internal::handle_zeros_in_scale(safe_range);

        const Scalar range_min = feature_range_.first;
        const Scalar range_max = feature_range_.second;
        scale_ = (range_max - range_min) / safe_range.array();
        min_ = RowVectorType(range_min - (data_min_.array() * scale_.array()));

        n_samples_seen_ += X.rows();
        return *this;
    }

    /// @brief Scale features of X according to `feature_range`.
    ///
    /// @param X Data matrix of shape (n_samples, n_features).
    /// @return Transformed data of same shape.
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] MatrixType transform_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        MatrixType result = (X.array().rowwise() * scale_.array()).rowwise() + min_.array();
        if (clip_) {
            result = result.cwiseMax(feature_range_.first).cwiseMin(feature_range_.second);
        }
        return result;
    }

    /// @brief Undo the scaling of X according to `feature_range`.
    ///
    /// @param X Transformed data of shape (n_samples, n_features).
    /// @return Un-transformed data of same shape.
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] MatrixType inverse_transform_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        return (X.rowwise() - min_).array().rowwise() / scale_.array();
    }

    /// @brief Transform features in-place to the target range.
    /// @param X Data matrix of shape (n_samples, n_features), modified in place.
    /// @throws std::runtime_error if the model has not been fitted.
    void transform_inplace(Eigen::Ref<MatrixType> X) const {
        this->check_is_fitted();
        this->validate_feature_count(X);
        X.array().rowwise() *= scale_.array();
        X.rowwise() += min_;
        if (clip_) {
            X = X.cwiseMax(feature_range_.first).cwiseMin(feature_range_.second);
        }
    }

    /// @brief Inverse-transform features in-place to original scale.
    /// @param X Scaled data matrix, modified in place.
    /// @throws std::runtime_error if the model has not been fitted.
    void inverse_transform_inplace(Eigen::Ref<MatrixType> X) const {
        this->check_is_fitted();
        this->validate_feature_count(X);
        X.rowwise() -= min_;
        X.array().rowwise() /= scale_.array();
    }

private:
    std::pair<Scalar, Scalar> feature_range_;
    bool clip_;

    RowVectorType data_min_;
    RowVectorType data_max_;
    RowVectorType data_range_;
    RowVectorType scale_;
    RowVectorType min_;
    IndexType n_samples_seen_ = 0;
};

/// @}

} // namespace Skigen

#endif // SKIGEN_PREPROCESSING_MIN_MAX_SCALER_H
