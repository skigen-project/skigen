// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_PREPROCESSING_MAX_ABS_SCALER_H
#define SKIGEN_PREPROCESSING_MAX_ABS_SCALER_H

#include "../Core/Base.h"
#include "../Core/EigenHelpers.h"
#include "../Core/Validation.h"

#include <Eigen/Core>

namespace Skigen {

template <typename Scalar = double>
class MaxAbsScaler
    : public Transformer<MaxAbsScaler<Scalar>, Scalar> {
public:
    using Base = Transformer<MaxAbsScaler<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;

    explicit MaxAbsScaler(bool clip = false) : clip_(clip) {}

    // -- Accessors ----------------------------------------------------------

    [[nodiscard]] bool clip() const noexcept { return clip_; }

    [[nodiscard]] const RowVectorType& max_abs() const {
        this->check_is_fitted(); return max_abs_;
    }
    [[nodiscard]] const RowVectorType& scale() const {
        this->check_is_fitted(); return scale_;
    }
    [[nodiscard]] IndexType n_samples_seen() const {
        this->check_is_fitted(); return n_samples_seen_;
    }

    // -- Implementation (called by CRTP base) --------------------------------

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

    [[nodiscard]] MatrixType transform_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        MatrixType result = X.array().rowwise() / scale_.array();
        if (clip_) {
            result = result.cwiseMax(Scalar{-1}).cwiseMin(Scalar{1});
        }
        return result;
    }

    [[nodiscard]] MatrixType inverse_transform_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        return X.array().rowwise() * scale_.array();
    }

    void transform_inplace(Eigen::Ref<MatrixType> X) const {
        this->check_is_fitted();
        this->validate_feature_count(X);
        X.array().rowwise() /= scale_.array();
        if (clip_) {
            X = X.cwiseMax(Scalar{-1}).cwiseMin(Scalar{1});
        }
    }

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

} // namespace Skigen

#endif // SKIGEN_PREPROCESSING_MAX_ABS_SCALER_H
