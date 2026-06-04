// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_PREPROCESSING_NORMALIZER_H
#define SKIGEN_PREPROCESSING_NORMALIZER_H

#include "../Core/Base.h"
#include "../Core/Validation.h"

#include <Eigen/Core>
#include <Eigen/SparseCore>
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
/// ### Limitations relative to scikit-learn
///
/// The following scikit-learn constructor
///   parameters are not honoured: `copy`.
///   The following sklearn fitted attributes are not exposed:
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

    // Make the dense base-class fit/transform overloads visible alongside
    // the sparse overloads added below.
    using Base::fit;
    using Base::transform;
    using Base::fit_transform;

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

    // -- Sparse-aware overloads --------------------------------

    /// @brief Fit on a sparse matrix (stateless — only records `n_features_in_`).
    template <int Options, typename StorageIndex>
    Normalizer& fit(
        const Eigen::SparseMatrix<Scalar, Options, StorageIndex>& X) {
        if (X.rows() == 0 || X.cols() == 0) {
            throw std::invalid_argument(
                "Normalizer.fit: empty sparse matrix.");
        }
        this->n_features_in_ = X.cols();
        this->fitted_ = true;
        return *this;
    }

    /// @brief Normalize each row of a sparse matrix to unit norm without
    ///   densifying.
    ///
    /// Computes the row norm by iterating over the explicit nonzeros of each
    /// row (RowMajor) or column (ColMajor → converted), then divides each
    /// stored value by that norm. Implicit zeros remain implicit.
    template <int Options, typename StorageIndex>
    [[nodiscard]] Eigen::SparseMatrix<Scalar, Options, StorageIndex>
    transform(
        const Eigen::SparseMatrix<Scalar, Options, StorageIndex>& X) const {
        this->check_is_fitted();
        if (X.cols() != this->n_features_in_) {
            throw std::invalid_argument(
                "X has " + std::to_string(X.cols()) +
                " features, but Normalizer was fitted with " +
                std::to_string(this->n_features_in_) + " features.");
        }
        // Operate on a row-major copy so we can iterate by row.
        using RowSparse =
            Eigen::SparseMatrix<Scalar, Eigen::RowMajor, StorageIndex>;
        RowSparse Xr = X;

        for (Eigen::Index i = 0; i < Xr.rows(); ++i) {
            Scalar nrm{0};
            switch (norm_) {
                case Norm::L1:
                    for (typename RowSparse::InnerIterator it(Xr, i); it; ++it)
                        nrm += std::abs(it.value());
                    break;
                case Norm::L2:
                    for (typename RowSparse::InnerIterator it(Xr, i); it; ++it)
                        nrm += it.value() * it.value();
                    nrm = std::sqrt(nrm);
                    break;
                case Norm::Max:
                    for (typename RowSparse::InnerIterator it(Xr, i); it; ++it)
                        nrm = std::max(nrm, std::abs(it.value()));
                    break;
            }
            if (nrm > Scalar{0}) {
                for (typename RowSparse::InnerIterator it(Xr, i); it; ++it) {
                    it.valueRef() = it.value() / nrm;
                }
            }
        }
        return Eigen::SparseMatrix<Scalar, Options, StorageIndex>(Xr);
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
