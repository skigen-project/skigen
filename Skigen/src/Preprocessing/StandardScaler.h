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

    explicit StandardScaler(bool with_mean = true, bool with_std = true)
        : with_mean_(with_mean), with_std_(with_std) {}

    // -- Accessors ----------------------------------------------------------

    [[nodiscard]] bool with_mean()  const noexcept { return with_mean_; }
    [[nodiscard]] bool with_std()   const noexcept { return with_std_; }

    [[nodiscard]] const RowVectorType& mean() const {
        this->check_is_fitted();
        return mean_;
    }

    [[nodiscard]] const RowVectorType& var() const {
        this->check_is_fitted();
        return var_;
    }

    [[nodiscard]] const RowVectorType& scale() const {
        this->check_is_fitted();
        return scale_;
    }

    [[nodiscard]] IndexType n_samples_seen() const {
        this->check_is_fitted();
        return n_samples_seen_;
    }

    // -- Implementation (called by CRTP base) --------------------------------

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

        if (!with_mean_) {
            // Keep computed mean for variance, but zero it for transform
            mean_.setZero();
        }

        this->fitted_ = true;
        return *this;
    }

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
    IndexType n_samples_seen_ = 0;
};

} // namespace Skigen

#endif // SKIGEN_PREPROCESSING_STANDARD_SCALER_H
