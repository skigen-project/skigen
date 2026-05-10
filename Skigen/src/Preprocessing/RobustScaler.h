// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_PREPROCESSING_ROBUST_SCALER_H
#define SKIGEN_PREPROCESSING_ROBUST_SCALER_H

#include "../Core/Base.h"
#include "../Core/EigenHelpers.h"
#include "../Core/Validation.h"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace Skigen {
namespace internal {

// Quantile computation for a sorted range (linear interpolation, numpy default)
template <typename Scalar>
Scalar quantile_sorted(const Scalar* data, Eigen::Index n, Scalar q) {
    if (n == 1) return data[0];
    const Scalar idx = q * static_cast<Scalar>(n - 1);
    const Eigen::Index lo = static_cast<Eigen::Index>(std::floor(idx));
    const Eigen::Index hi = static_cast<Eigen::Index>(std::ceil(idx));
    if (lo == hi) return data[lo];
    const Scalar frac = idx - static_cast<Scalar>(lo);
    return data[lo] * (Scalar{1} - frac) + data[hi] * frac;
}

} // namespace internal

/// @defgroup Algo_RobustScaler RobustScaler
/// @ingroup Preprocessing
/// @brief Scale features using statistics that are robust to outliers.
/// @{

/// @brief Scale features using statistics that are robust to outliers.
///
/// This Scaler removes the median and scales the data according to
/// the quantile range (defaults to IQR: Interquartile Range). The IQR
/// is the range between the 1st quartile (25th quantile) and the 3rd
/// quartile (75th quantile).
///
/// Centering and scaling happen independently on each feature.
///
/// Mirrors
/// [sklearn.preprocessing.RobustScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html).
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `with_centering` | `bool` | `true` | If `true`, center the data before scaling (subtract median). |
/// | `with_scaling` | `bool` | `true` | If `true`, scale the data to interquartile range. |
/// | `quantile_range` | `std::pair<Scalar,Scalar>` | `{25, 75}` | Quantile range used to calculate `scale_`. Must satisfy `0 ≤ q_min < q_max ≤ 100`. |
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type | Description |
/// |----------|------|-------------|
/// | `center()` | `RowVectorType` | Per-feature median (1 × n_features). |
/// | `scale()` | `RowVectorType` | Per-feature interquartile range (1 × n_features). |
///
/// ### See also
///
/// - Skigen::StandardScaler — Standardize using mean and std (not robust to outliers).
/// - Skigen::MinMaxScaler — Scale features to a given range.
///
/// ### Limitations relative to scikit-learn
///
/// The following scikit-learn constructor
///   parameters are not honoured: `copy`, `unit_variance`.
///   The following sklearn fitted attributes are not exposed:
///   `n_features_in_`, `feature_names_in_`.
///
/// ### Examples
///
/// @snippet robust_scaler.cpp example_robust_scaler
template <typename Scalar = double>
class RobustScaler
    : public Transformer<RobustScaler<Scalar>, Scalar> {
public:
    using Base = Transformer<RobustScaler<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;

    /// @brief Construct a RobustScaler.
    ///
    /// @param with_centering If `true`, center the data before scaling
    ///   by subtracting the median (`bool`, default `true`).
    /// @param with_scaling If `true`, scale the data to interquartile range
    ///   (`bool`, default `true`).
    /// @param quantile_range Quantile range for computing `scale_`
    ///   (`std::pair<Scalar,Scalar>`, default `{25, 75}`).
    /// @throws std::invalid_argument if `quantile_range` is invalid.
    explicit RobustScaler(bool with_centering = true,
                          bool with_scaling = true,
                          std::pair<Scalar, Scalar> quantile_range = {Scalar{25}, Scalar{75}})
        : with_centering_(with_centering),
          with_scaling_(with_scaling),
          quantile_range_(quantile_range) {
        if (quantile_range_.first < Scalar{0} || quantile_range_.second > Scalar{100} ||
            quantile_range_.first >= quantile_range_.second) {
            throw std::invalid_argument(
                "quantile_range must satisfy 0 <= q_min < q_max <= 100.");
        }
    }

    // -- Accessors ----------------------------------------------------------

    /// @brief Whether centering is enabled.
    [[nodiscard]] bool with_centering() const noexcept { return with_centering_; }
    /// @brief Whether scaling is enabled.
    [[nodiscard]] bool with_scaling() const noexcept { return with_scaling_; }

    /// @brief Per-feature median (1 × n_features).
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] const RowVectorType& center() const {
        this->check_is_fitted(); return center_;
    }
    /// @brief Per-feature interquartile range (1 × n_features).
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] const RowVectorType& scale() const {
        this->check_is_fitted(); return scale_;
    }

    // -- Implementation (called by CRTP base) --------------------------------

    /// @brief Compute median and quantile range for later scaling.
    ///
    /// @param X Training data of shape (n_samples, n_features).
    /// @return Reference to the fitted transformer (`*this`).
    RobustScaler& fit_impl(const Eigen::Ref<const MatrixType>& X) {
        internal::check_non_empty(X);

        this->n_features_in_ = X.cols();
        const Eigen::Index n = X.rows();
        const Eigen::Index p = X.cols();

        const Scalar q_lo = quantile_range_.first / Scalar{100};
        const Scalar q_hi = quantile_range_.second / Scalar{100};

        center_ = RowVectorType::Zero(p);
        scale_ = RowVectorType::Ones(p);

        // Sort each column to compute median and quantiles
        std::vector<Scalar> col_buf(static_cast<std::size_t>(n));

        for (Eigen::Index j = 0; j < p; ++j) {
            for (Eigen::Index i = 0; i < n; ++i) {
                col_buf[static_cast<std::size_t>(i)] = X(i, j);
            }
            std::sort(col_buf.begin(), col_buf.end());

            if (with_centering_) {
                center_(j) = internal::quantile_sorted<Scalar>(
                    col_buf.data(), n, Scalar{0.5});
            }
            if (with_scaling_) {
                const Scalar lo = internal::quantile_sorted<Scalar>(
                    col_buf.data(), n, q_lo);
                const Scalar hi = internal::quantile_sorted<Scalar>(
                    col_buf.data(), n, q_hi);
                scale_(j) = hi - lo;
            }
        }

        if (with_scaling_) {
            internal::handle_zeros_in_scale(scale_);
        }

        this->fitted_ = true;
        return *this;
    }

    /// @brief Center and scale the data.
    ///
    /// @param X Data matrix of shape (n_samples, n_features).
    /// @return Transformed data of same shape.
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] MatrixType transform_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        MatrixType result = X;
        if (with_centering_) {
            result.rowwise() -= center_;
        }
        if (with_scaling_) {
            result.array().rowwise() /= scale_.array();
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
        MatrixType result = X;
        if (with_scaling_) {
            result.array().rowwise() *= scale_.array();
        }
        if (with_centering_) {
            result.rowwise() += center_;
        }
        return result;
    }

    /// @brief Transform features in-place using robust statistics.
    /// @param X Data matrix of shape (n_samples, n_features), modified in place.
    /// @throws std::runtime_error if the model has not been fitted.
    void transform_inplace(Eigen::Ref<MatrixType> X) const {
        this->check_is_fitted();
        this->validate_feature_count(X);
        if (with_centering_) X.rowwise() -= center_;
        if (with_scaling_) X.array().rowwise() /= scale_.array();
    }

    /// @brief Inverse-transform features in-place to original scale.
    /// @param X Scaled data matrix, modified in place.
    /// @throws std::runtime_error if the model has not been fitted.
    void inverse_transform_inplace(Eigen::Ref<MatrixType> X) const {
        this->check_is_fitted();
        this->validate_feature_count(X);
        if (with_scaling_) X.array().rowwise() *= scale_.array();
        if (with_centering_) X.rowwise() += center_;
    }

private:
    bool with_centering_;
    bool with_scaling_;
    std::pair<Scalar, Scalar> quantile_range_;

    RowVectorType center_;
    RowVectorType scale_;
};

/// @}

} // namespace Skigen

#endif // SKIGEN_PREPROCESSING_ROBUST_SCALER_H
