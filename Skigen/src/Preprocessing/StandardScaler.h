// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_PREPROCESSING_STANDARD_SCALER_H
#define SKIGEN_PREPROCESSING_STANDARD_SCALER_H

#include "../Core/Base.h"
#include "../Core/EigenHelpers.h"
#include "../Core/Validation.h"

#include <Eigen/Core>
#include <cmath>
#include <limits>

namespace Skigen {

/// @defgroup Algo_StandardScaler StandardScaler
/// @ingroup Preprocessing
/// @brief Standardize features by removing the mean and scaling to unit variance.
/// @{

/// @brief Standardize features by removing the mean and scaling to unit variance.
///
/// The standard score of a sample `x` is calculated as:
///
/// @f[
///   z = \frac{x - \mu}{\sigma}
/// @f]
///
/// where @f$\mu@f$ is the mean of the training samples and @f$\sigma@f$
/// is the standard deviation. Centering and scaling happen independently
/// on each feature by computing the relevant statistics on the samples in
/// the training set.
///
/// Mirrors
/// [sklearn.preprocessing.StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `with_mean` | `bool` | `true` | If `true`, center the data before scaling. |
/// | `with_std` | `bool` | `true` | If `true`, scale the data to unit variance. |
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type | Description |
/// |----------|------|-------------|
/// | `mean()` | `RowVectorType` | Per-feature mean of shape (1 × n_features). |
/// | `var()` | `RowVectorType` | Per-feature variance of shape (1 × n_features). |
/// | `scale()` | `RowVectorType` | Per-feature standard deviation (1 × n_features). |
/// | `n_samples_seen()` | `IndexType` | Number of samples processed by `fit()`. |
///
/// ### See also
///
/// - Skigen::MinMaxScaler — Scale features to a given range.
/// - Skigen::RobustScaler — Scale using statistics robust to outliers.
/// - Skigen::MaxAbsScaler — Scale each feature by its maximum absolute value.
///
/// @note **scikit-learn parity gaps:** The following sklearn constructor
///   parameters are not yet supported: `copy`.
///   `partial_fit()` is supported via Chan's online algorithm (sklearn parity).
///   The following sklearn fitted attributes are not yet exposed:
///   `n_features_in_`, `feature_names_in_`.
///
/// ### Examples
///
/// @snippet standard_scaler.cpp example_standard_scaler
template <typename Scalar = double>
class StandardScaler
    : public Transformer<StandardScaler<Scalar>, Scalar> {
public:
    using Base = Transformer<StandardScaler<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;
    using typename Base::ScalarType;

    /// @brief Construct a StandardScaler.
    ///
    /// @param with_mean If `true`, center the data before scaling (`bool`, default `true`).
    /// @param with_std If `true`, scale the data to unit variance (`bool`, default `true`).
    explicit StandardScaler(bool with_mean = true, bool with_std = true)
        : with_mean_(with_mean), with_std_(with_std) {}

    // -- Accessors ----------------------------------------------------------

    /// @brief Whether centering is enabled.
    [[nodiscard]] bool with_mean()  const noexcept { return with_mean_; }
    /// @brief Whether scaling to unit variance is enabled.
    [[nodiscard]] bool with_std()   const noexcept { return with_std_; }

    /// @brief Per-feature mean (1 × n_features).
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] const RowVectorType& mean() const {
        this->check_is_fitted();
        return mean_;
    }

    /// @brief Per-feature variance (1 × n_features).
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] const RowVectorType& var() const {
        this->check_is_fitted();
        return var_;
    }

    /// @brief Per-feature standard deviation (1 × n_features).
    ///
    /// Used for scaling; zeros are replaced with 1 to avoid division by zero.
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] const RowVectorType& scale() const {
        this->check_is_fitted();
        return scale_;
    }

    /// @brief Number of samples processed during `fit()`.
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] IndexType n_samples_seen() const {
        this->check_is_fitted();
        return n_samples_seen_;
    }

    // -- Implementation (called by CRTP base) --------------------------------

    /// @brief Compute the mean and std to be used for later scaling.
    ///
    /// @param X Training data of shape (n_samples, n_features).
    /// @return Reference to the fitted transformer (`*this`).
    StandardScaler& fit_impl(const Eigen::Ref<const MatrixType>& X) {
        internal::check_non_empty(X);

        this->n_features_in_ = X.cols();
        n_samples_seen_ = X.rows();

        // Always compute mean (needed for variance as well)
        mean_ = X.colwise().mean();

        if (with_std_) {
            var_ = internal::colwise_variance<Scalar>(X, mean_);
            scale_ = var_.array().sqrt();
            internal::handle_zeros_in_scale(scale_);
        } else {
            var_ = RowVectorType::Ones(X.cols());
            scale_ = RowVectorType::Ones(X.cols());
        }

        // Track sum of squared deviations so partial_fit can extend the
        // statistics using a Welford / Chan combined update without losing
        // numerical precision.
        if (with_std_) {
            M2_ = var_ * static_cast<Scalar>(X.rows());
        } else {
            M2_ = RowVectorType::Zero(X.cols());
        }

        // mean_ stores the true sample mean even when `with_mean_=false`,
        // matching sklearn's StandardScaler.mean_ attribute. transform()
        // branches on `with_mean_` to decide whether to subtract it.
        this->fitted_ = true;
        return *this;
    }

    /// @brief Online update of the mean and variance using Chan's parallel
    ///   algorithm (a numerically-stable Welford variant).
    ///
    /// `partial_fit(X1).partial_fit(X2)` produces the same fitted attributes
    /// (within floating-point tolerance) as `fit([X1; X2])`, matching
    /// sklearn's `StandardScaler.partial_fit` contract.
    ///
    /// @param X Batch of training data, shape (n_samples_batch, n_features).
    ///   The feature count must match the first batch.
    /// @return Reference to the fitted transformer (`*this`).
    /// @throws std::invalid_argument if the feature count differs from the
    ///   first batch, or if X is empty.
    StandardScaler& partial_fit(const Eigen::Ref<const MatrixType>& X) {
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

        const Scalar n_old = static_cast<Scalar>(n_samples_seen_);
        const Scalar n_batch = static_cast<Scalar>(X.rows());
        const Scalar n_new = n_old + n_batch;

        const RowVectorType batch_mean = X.colwise().mean();
        const RowVectorType delta = batch_mean - mean_;

        // mean_new = mean_old + delta * (n_batch / n_new) — Chan's update
        mean_ = (mean_.array() +
                 delta.array() * (n_batch / n_new)).matrix();

        if (with_std_) {
            const RowVectorType batch_var =
                internal::colwise_variance<Scalar>(X, batch_mean);
            const RowVectorType batch_M2 = batch_var * n_batch;
            const Scalar mix = n_old * n_batch / n_new;
            M2_ = (M2_.array() + batch_M2.array() +
                   delta.array().square() * mix).matrix();
            var_ = M2_ / n_new;
            scale_ = var_.array().sqrt();
            internal::handle_zeros_in_scale(scale_);
        }

        n_samples_seen_ = static_cast<IndexType>(n_new);
        return *this;
    }

    /// @brief Perform standardization by centering and scaling.
    ///
    /// @param X Data matrix of shape (n_samples, n_features).
    /// @return Transformed data of same shape.
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] MatrixType transform_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        if (with_mean_ && with_std_) {
            return (X.rowwise() - mean_).array().rowwise() / scale_.array();
        } else if (with_mean_) {
            return X.rowwise() - mean_;
        } else if (with_std_) {
            return X.array().rowwise() / scale_.array();
        }
        return X;
    }

    /// @brief Scale back the data to the original representation.
    ///
    /// @param X Transformed data of shape (n_samples, n_features).
    /// @return Un-transformed data of same shape.
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] MatrixType inverse_transform_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        if (with_mean_ && with_std_) {
            return (X.array().rowwise() * scale_.array()).matrix().rowwise() +
                   mean_;
        } else if (with_mean_) {
            return X.rowwise() + mean_;
        } else if (with_std_) {
            return X.array().rowwise() * scale_.array();
        }
        return X;
    }

    // -- EES extension: in-place transform -----------------------------------

    /// @brief Transform features in-place by standardizing.
    /// @param X Data matrix of shape (n_samples, n_features), modified in place.
    /// @throws std::runtime_error if the model has not been fitted.
    void transform_inplace(Eigen::Ref<MatrixType> X) const {
        this->check_is_fitted();
        this->validate_feature_count(X);

        if (with_mean_) {
            X.rowwise() -= mean_;
        }
        if (with_std_) {
            X.array().rowwise() /= scale_.array();
        }
    }

    /// @brief Inverse-transform features in-place.
    /// @param X Standardized data matrix, modified in place to original scale.
    /// @throws std::runtime_error if the model has not been fitted.
    void inverse_transform_inplace(Eigen::Ref<MatrixType> X) const {
        this->check_is_fitted();
        this->validate_feature_count(X);

        if (with_std_) {
            X.array().rowwise() *= scale_.array();
        }
        if (with_mean_) {
            X.rowwise() += mean_;
        }
    }

private:
    bool with_mean_;
    bool with_std_;

    RowVectorType mean_;
    RowVectorType var_;
    RowVectorType scale_;
    RowVectorType M2_;          ///< Running sum of squared deviations for partial_fit.
    IndexType n_samples_seen_ = 0;
};

/// @}

} // namespace Skigen

#endif // SKIGEN_PREPROCESSING_STANDARD_SCALER_H
