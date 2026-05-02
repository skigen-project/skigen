// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_PREPROCESSING_NORMALIZER_H
#define SKIGEN_PREPROCESSING_NORMALIZER_H

#include "../Core/Base.h"
#include "../Core/Validation.h"

#include <Eigen/Core>
#include <cmath>
#include <stdexcept>
#include <string>

namespace Skigen {

/// @brief Norm type used by Normalizer.
enum class Norm { L1, L2, Max };

/// @defgroup Algo_Normalizer Normalizer
/// @ingroup Preprocessing
/// @brief Normalize samples individually to unit norm.
/// @{

/// @brief Normalize samples individually to unit norm.
///
/// Each sample (i.e. each row of the data matrix) with at least one
/// non-zero component is rescaled independently of other samples so
/// that its norm (L1, L2, or max) equals one.
///
/// Mirrors
/// [sklearn.preprocessing.Normalizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html).
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `norm` | `Norm` | `Norm::L2` | The norm to use: `Norm::L1`, `Norm::L2`, or `Norm::Max`. |
///
/// ### Notes
///
/// This transformer is stateless — `fit()` only records the number of
/// features. `inverse_transform()` is not supported because
/// normalization is not reversible.
///
/// @note **scikit-learn parity gaps:** The following sklearn constructor
///   parameters are not yet supported: `copy`.
///   The following sklearn fitted attributes are not yet exposed:
///   `n_features_in_`, `feature_names_in_`.
///
/// ### Examples
///
/// @snippet normalizer.cpp example_normalizer
template <typename Scalar = double>
class Normalizer
    : public Transformer<Normalizer<Scalar>, Scalar> {
public:
    using Base = Transformer<Normalizer<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::IndexType;

    /// @brief Construct a Normalizer.
    ///
    /// @param norm The norm to use (`Norm`, default `Norm::L2`).
    explicit Normalizer(Norm norm = Norm::L2) : norm_(norm) {}

    // -- Accessors ----------------------------------------------------------

    /// @brief The norm type in use.
    [[nodiscard]] Norm norm() const noexcept { return norm_; }

    // -- Implementation (called by CRTP base) --------------------------------

    /// @brief Fit the Normalizer (stateless — only records `n_features_in_`).
    ///
    /// @param X Training data of shape (n_samples, n_features).
    /// @return Reference to the fitted transformer (`*this`).
    // Normalizer is stateless — fit only records n_features_in_
    Normalizer& fit_impl(const Eigen::Ref<const MatrixType>& X) {
        internal::check_non_empty(X);
        this->n_features_in_ = X.cols();
        this->fitted_ = true;
        return *this;
    }

    /// @brief Normalize each sample to unit norm.
    ///
    /// @param X Data matrix of shape (n_samples, n_features).
    /// @return Normalized data of same shape.
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] MatrixType transform_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        MatrixType result = X;
        normalize_rows(result);
        return result;
    }

    /// @brief Not supported — normalization is not reversible.
    ///
    /// @throws std::runtime_error Always throws.
    // Normalization is not invertible
    [[nodiscard]] MatrixType inverse_transform_impl(
        const Eigen::Ref<const MatrixType>& /*X*/) const {
        throw std::runtime_error(
            "Normalizer does not support inverse_transform. "
            "Row normalization is not reversible.");
    }

    /// @brief Normalize samples in-place to unit norm.
    /// @param X Data matrix of shape (n_samples, n_features), modified in place.
    /// @throws std::runtime_error if the model has not been fitted.
    void transform_inplace(Eigen::Ref<MatrixType> X) const {
        this->check_is_fitted();
        this->validate_feature_count(X);
        normalize_rows(X);
    }

private:
    Norm norm_;

    template <typename Derived>
    void normalize_rows(Eigen::MatrixBase<Derived>& X_base) const {
        auto& X = X_base.derived();
        for (Eigen::Index i = 0; i < X.rows(); ++i) {
            Scalar nrm{};
            switch (norm_) {
                case Norm::L1:
                    nrm = X.row(i).cwiseAbs().sum();
                    break;
                case Norm::L2:
                    nrm = X.row(i).norm();
                    break;
                case Norm::Max:
                    nrm = X.row(i).cwiseAbs().maxCoeff();
                    break;
            }
            if (nrm > Scalar{0}) {
                X.row(i) /= nrm;
            }
        }
    }
};

/// @}

} // namespace Skigen

#endif // SKIGEN_PREPROCESSING_NORMALIZER_H
