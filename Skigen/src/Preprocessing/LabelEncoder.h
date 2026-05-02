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

/// @defgroup Algo_LabelEncoder LabelEncoder
/// @ingroup Preprocessing
/// @brief Encode integer labels as contiguous indices [0, n_classes).
/// @{

/// @brief Encode target labels with value between 0 and n_classes − 1.
///
/// This transformer encodes integer labels as contiguous zero-based
/// indices. It is useful as a utility to normalize labels.
///
/// Mirrors
/// [sklearn.preprocessing.LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
/// for integer-typed labels.
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type | Description |
/// |----------|------|-------------|
/// | `classes()` | `std::vector<Label>` | Unique class labels sorted in ascending order. |
/// | `n_classes()` | `IndexType` | Number of unique classes. |
///
/// ### Notes
///
/// Unlike the sklearn version which supports string labels, this
/// implementation only supports integer (or any arithmetic) label types.
///
/// @note **scikit-learn parity gaps:** String labels are not supported;
///   only integer or arithmetic label types are handled.
///
/// ### Examples
///
/// @snippet label_encoder.cpp example_label_encoder
template <typename Label = int>
class LabelEncoder {
public:
    using VectorType = Eigen::Matrix<Label, Eigen::Dynamic, 1>;
    using IndexVector = Eigen::VectorXi;

    // -- Accessors ----------------------------------------------------------

    /// @brief Whether the encoder has been fitted.
    [[nodiscard]] bool is_fitted() const noexcept { return fitted_; }

    /// @brief Unique class labels sorted in ascending order.
    ///
    /// @return Read-only reference to the class vector.
    /// @throws std::runtime_error if the encoder has not been fitted.
    [[nodiscard]] const std::vector<Label>& classes() const {
        if (!fitted_) throw std::runtime_error(
            "LabelEncoder has not been fitted yet.");
        return classes_;
    }

    /// @brief Number of unique classes.
    /// @throws std::runtime_error if the encoder has not been fitted.
    [[nodiscard]] Eigen::Index n_classes() const {
        if (!fitted_) throw std::runtime_error(
            "LabelEncoder has not been fitted yet.");
        return static_cast<Eigen::Index>(classes_.size());
    }

    // -- fit / transform / inverse_transform --------------------------------

    /// @brief Fit the label encoder by discovering unique classes.
    ///
    /// @param y Label vector of shape (n_samples,).
    /// @return Reference to the fitted encoder (`*this`).
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

    /// @brief Transform labels to normalized encoding.
    ///
    /// @param y Label vector of shape (n_samples,).
    /// @return Integer vector of encoded labels [0, n_classes).
    /// @throws std::runtime_error if the encoder has not been fitted.
    /// @throws std::invalid_argument if `y` contains unseen labels.
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

    /// @brief Fit the encoder and return transformed labels.
    ///
    /// Equivalent to `fit(y).transform(y)` but more convenient.
    ///
    /// @param y Label vector of shape (n_samples,).
    /// @return Integer vector of encoded labels [0, n_classes).
    [[nodiscard]] IndexVector fit_transform(
        const Eigen::Ref<const VectorType>& y) {
        fit(y);
        return transform(y);
    }

    /// @brief Transform encoded labels back to original labels.
    ///
    /// @param y Integer vector of encoded labels.
    /// @return Original label vector of shape (n_samples,).
    /// @throws std::runtime_error if the encoder has not been fitted.
    /// @throws std::invalid_argument if `y` contains indices out of range.
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

/// @}

} // namespace Skigen

#endif // SKIGEN_PREPROCESSING_LABEL_ENCODER_H
