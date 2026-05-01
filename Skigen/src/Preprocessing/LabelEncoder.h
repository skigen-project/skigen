// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_PREPROCESSING_LABEL_ENCODER_H
#define SKIGEN_PREPROCESSING_LABEL_ENCODER_H

#include "../Core/Validation.h"

#include <Eigen/Core>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace Skigen {

/// LabelEncoder — encode integer labels as contiguous indices [0, n_classes).
/// Mirrors sklearn.preprocessing.LabelEncoder for integer-typed labels.
template <typename Label = int>
class LabelEncoder {
public:
    using VectorType = Eigen::Matrix<Label, Eigen::Dynamic, 1>;
    using IndexVector = Eigen::VectorXi;

    // -- Accessors ----------------------------------------------------------

    [[nodiscard]] bool is_fitted() const noexcept { return fitted_; }

    [[nodiscard]] const std::vector<Label>& classes() const {
        if (!fitted_) throw std::runtime_error(
            "LabelEncoder has not been fitted yet.");
        return classes_;
    }

    [[nodiscard]] Eigen::Index n_classes() const {
        if (!fitted_) throw std::runtime_error(
            "LabelEncoder has not been fitted yet.");
        return static_cast<Eigen::Index>(classes_.size());
    }

    // -- fit / transform / inverse_transform --------------------------------

    LabelEncoder& fit(const Eigen::Ref<const VectorType>& y) {
        classes_.clear();
        classes_.reserve(static_cast<std::size_t>(y.size()));
        for (Eigen::Index i = 0; i < y.size(); ++i) {
            classes_.push_back(y(i));
        }
        std::sort(classes_.begin(), classes_.end());
        classes_.erase(std::unique(classes_.begin(), classes_.end()),
                       classes_.end());
        fitted_ = true;
        return *this;
    }

    [[nodiscard]] IndexVector transform(
        const Eigen::Ref<const VectorType>& y) const {
        if (!fitted_) throw std::runtime_error(
            "LabelEncoder has not been fitted yet.");

        IndexVector encoded(y.size());
        for (Eigen::Index i = 0; i < y.size(); ++i) {
            auto it = std::lower_bound(classes_.begin(), classes_.end(), y(i));
            if (it == classes_.end() || *it != y(i)) {
                throw std::invalid_argument(
                    "y contains previously unseen label.");
            }
            encoded(i) = static_cast<int>(std::distance(classes_.begin(), it));
        }
        return encoded;
    }

    [[nodiscard]] IndexVector fit_transform(
        const Eigen::Ref<const VectorType>& y) {
        fit(y);
        return transform(y);
    }

    [[nodiscard]] VectorType inverse_transform(
        const Eigen::Ref<const IndexVector>& y) const {
        if (!fitted_) throw std::runtime_error(
            "LabelEncoder has not been fitted yet.");

        VectorType decoded(y.size());
        const auto n = static_cast<int>(classes_.size());
        for (Eigen::Index i = 0; i < y.size(); ++i) {
            if (y(i) < 0 || y(i) >= n) {
                throw std::invalid_argument(
                    "y contains index out of range [0, n_classes).");
            }
            decoded(i) = classes_[static_cast<std::size_t>(y(i))];
        }
        return decoded;
    }

private:
    bool fitted_ = false;
    std::vector<Label> classes_;
};

} // namespace Skigen

#endif // SKIGEN_PREPROCESSING_LABEL_ENCODER_H
