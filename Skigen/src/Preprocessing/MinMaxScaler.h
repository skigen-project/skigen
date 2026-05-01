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

template <typename Scalar = double>
class MinMaxScaler
    : public Transformer<MinMaxScaler<Scalar>, Scalar> {
public:
    using Base = Transformer<MinMaxScaler<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;

    explicit MinMaxScaler(std::pair<Scalar, Scalar> feature_range = {Scalar{0}, Scalar{1}},
                          bool clip = false)
        : feature_range_(feature_range), clip_(clip) {
        if (feature_range_.first >= feature_range_.second) {
            throw std::invalid_argument(
                "feature_range: min must be strictly less than max.");
        }
    }

    // -- Accessors ----------------------------------------------------------

    [[nodiscard]] const std::pair<Scalar, Scalar>& feature_range() const noexcept {
        return feature_range_;
    }
    [[nodiscard]] bool clip() const noexcept { return clip_; }

    [[nodiscard]] const RowVectorType& data_min() const {
        this->check_is_fitted(); return data_min_;
    }
    [[nodiscard]] const RowVectorType& data_max() const {
        this->check_is_fitted(); return data_max_;
    }
    [[nodiscard]] const RowVectorType& data_range() const {
        this->check_is_fitted(); return data_range_;
    }
    [[nodiscard]] const RowVectorType& scale() const {
        this->check_is_fitted(); return scale_;
    }
    [[nodiscard]] const RowVectorType& min() const {
        this->check_is_fitted(); return min_;
    }
    [[nodiscard]] IndexType n_samples_seen() const {
        this->check_is_fitted(); return n_samples_seen_;
    }

    // -- Implementation (called by CRTP base) --------------------------------

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

    [[nodiscard]] MatrixType transform_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        MatrixType result = (X.array().rowwise() * scale_.array()).rowwise() + min_.array();
        if (clip_) {
            result = result.cwiseMax(feature_range_.first).cwiseMin(feature_range_.second);
        }
        return result;
    }

    [[nodiscard]] MatrixType inverse_transform_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        return (X.rowwise() - min_).array().rowwise() / scale_.array();
    }

    void transform_inplace(Eigen::Ref<MatrixType> X) const {
        this->check_is_fitted();
        this->validate_feature_count(X);
        X.array().rowwise() *= scale_.array();
        X.rowwise() += min_;
        if (clip_) {
            X = X.cwiseMax(feature_range_.first).cwiseMin(feature_range_.second);
        }
    }

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

} // namespace Skigen

#endif // SKIGEN_PREPROCESSING_MIN_MAX_SCALER_H
