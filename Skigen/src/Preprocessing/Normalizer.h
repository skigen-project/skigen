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

enum class Norm { L1, L2, Max };

template <typename Scalar = double>
class Normalizer
    : public Transformer<Normalizer<Scalar>, Scalar> {
public:
    using Base = Transformer<Normalizer<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::IndexType;

    explicit Normalizer(Norm norm = Norm::L2) : norm_(norm) {}

    // -- Accessors ----------------------------------------------------------

    [[nodiscard]] Norm norm() const noexcept { return norm_; }

    // -- Implementation (called by CRTP base) --------------------------------

    // Normalizer is stateless — fit only records n_features_in_
    Normalizer& fit_impl(const Eigen::Ref<const MatrixType>& X) {
        internal::check_non_empty(X);
        this->n_features_in_ = X.cols();
        this->fitted_ = true;
        return *this;
    }

    [[nodiscard]] MatrixType transform_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        MatrixType result = X;
        normalize_rows(result);
        return result;
    }

    // Normalization is not invertible
    [[nodiscard]] MatrixType inverse_transform_impl(
        const Eigen::Ref<const MatrixType>& /*X*/) const {
        throw std::runtime_error(
            "Normalizer does not support inverse_transform. "
            "Row normalization is not reversible.");
    }

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

} // namespace Skigen

#endif // SKIGEN_PREPROCESSING_NORMALIZER_H
