// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_CORE_BASE_H
#define SKIGEN_CORE_BASE_H

#include "Params.h"
#include "Traits.h"

#include <Eigen/Core>
#include <stdexcept>
#include <string>
#include <string_view>

namespace Skigen {

// ---------------------------------------------------------------------------
// Estimator — CRTP base for all fitted objects
// ---------------------------------------------------------------------------

template <typename Derived, typename Scalar = double>
class Estimator : public EigenTypes<Scalar> {
public:
    using typename EigenTypes<Scalar>::MatrixType;
    using typename EigenTypes<Scalar>::VectorType;
    using typename EigenTypes<Scalar>::RowVectorType;
    using typename EigenTypes<Scalar>::IndexType;
    using typename EigenTypes<Scalar>::ScalarType;

    [[nodiscard]] bool is_fitted() const noexcept { return fitted_; }
    [[nodiscard]] IndexType n_features_in() const noexcept { return n_features_in_; }

    // Hyperparameter reflection — CRTP-dispatched into `set_param_impl` /
    // `get_params_impl` defined by each derived estimator, typically via
    // the SKIGEN_PARAMS(...) macro in <Skigen/src/Core/Params.h>.
    // Estimators that don't register any parameters get the default
    // behaviour: `set_param` throws `UnknownParameter` for any name and
    // `get_params` returns an empty dict.

    void set_param(std::string_view name, const ParameterValue& v) {
        if constexpr (
            requires(Derived& d) { d.set_param_impl(name, v); }) {
            static_cast<Derived*>(this)->set_param_impl(name, v);
        } else {
            throw UnknownParameter(name);
        }
    }
    [[nodiscard]] ParameterDict get_params() const {
        if constexpr (
            requires(const Derived& d) { d.get_params_impl(); }) {
            return static_cast<const Derived*>(this)->get_params_impl();
        } else {
            return ParameterDict{};
        }
    }

protected:
    bool fitted_ = false;
    IndexType n_features_in_ = 0;

    void check_is_fitted() const {
        if (!fitted_) {
            throw std::runtime_error("This estimator has not been fitted yet. "
                                     "Call fit() before using this method.");
        }
    }

    void validate_feature_count(const Eigen::Ref<const MatrixType>& X) const {
        if (X.cols() != n_features_in_) {
            throw std::invalid_argument(
                "X has " + std::to_string(X.cols()) + " features, but this "
                "estimator was fitted with " +
                std::to_string(n_features_in_) + " features.");
        }
    }
};

// ---------------------------------------------------------------------------
// Transformer — CRTP base for fit/transform/inverse_transform estimators
// ---------------------------------------------------------------------------

template <typename Derived, typename Scalar = double>
class Transformer : public Estimator<Derived, Scalar> {
public:
    using typename Estimator<Derived, Scalar>::MatrixType;
    using typename Estimator<Derived, Scalar>::VectorType;
    using typename Estimator<Derived, Scalar>::RowVectorType;
    using typename Estimator<Derived, Scalar>::IndexType;
    using typename Estimator<Derived, Scalar>::ScalarType;

    Derived& fit(const Eigen::Ref<const MatrixType>& X) {
        return static_cast<Derived*>(this)->fit_impl(X);
    }

    [[nodiscard]] MatrixType transform(
        const Eigen::Ref<const MatrixType>& X) const {
        this->check_is_fitted();
        this->validate_feature_count(X);
        return static_cast<const Derived*>(this)->transform_impl(X);
    }

    [[nodiscard]] MatrixType fit_transform(
        const Eigen::Ref<const MatrixType>& X) {
        fit(X);
        return static_cast<const Derived*>(this)->transform_impl(X);
    }

    [[nodiscard]] MatrixType inverse_transform(
        const Eigen::Ref<const MatrixType>& X) const {
        this->check_is_fitted();
        return static_cast<const Derived*>(this)->inverse_transform_impl(X);
    }
};

// ---------------------------------------------------------------------------
// Predictor — CRTP base for fit/predict/score estimators
// ---------------------------------------------------------------------------

template <typename Derived, typename Scalar = double>
class Predictor : public Estimator<Derived, Scalar> {
public:
    using typename Estimator<Derived, Scalar>::MatrixType;
    using typename Estimator<Derived, Scalar>::VectorType;
    using typename Estimator<Derived, Scalar>::RowVectorType;
    using typename Estimator<Derived, Scalar>::IndexType;
    using typename Estimator<Derived, Scalar>::ScalarType;

    Derived& fit(const Eigen::Ref<const MatrixType>& X,
                 const Eigen::Ref<const VectorType>& y) {
        return static_cast<Derived*>(this)->fit_impl(X, y);
    }

    [[nodiscard]] VectorType predict(
        const Eigen::Ref<const MatrixType>& X) const {
        this->check_is_fitted();
        this->validate_feature_count(X);
        return static_cast<const Derived*>(this)->predict_impl(X);
    }

    [[nodiscard]] ScalarType score(
        const Eigen::Ref<const MatrixType>& X,
        const Eigen::Ref<const VectorType>& y) const {
        this->check_is_fitted();
        this->validate_feature_count(X);
        return static_cast<const Derived*>(this)->score_impl(X, y);
    }
};

// ---------------------------------------------------------------------------
// Classifier — CRTP base for fit(X, y)/predict(X)/score(X, y) with integer labels
// ---------------------------------------------------------------------------

/// @brief CRTP base for classifiers that use integer (Eigen::VectorXi) labels.
///
/// Unlike Predictor (which uses floating-point target vectors), Classifier
/// operates on `Eigen::VectorXi` labels and returns mean accuracy as score.
template <typename Derived, typename Scalar = double>
class Classifier : public Estimator<Derived, Scalar> {
public:
    using typename Estimator<Derived, Scalar>::MatrixType;
    using typename Estimator<Derived, Scalar>::VectorType;
    using typename Estimator<Derived, Scalar>::RowVectorType;
    using typename Estimator<Derived, Scalar>::IndexType;
    using typename Estimator<Derived, Scalar>::ScalarType;
    using LabelType = Eigen::VectorXi;

    Derived& fit(const Eigen::Ref<const MatrixType>& X,
                 const Eigen::Ref<const LabelType>& y) {
        return static_cast<Derived*>(this)->fit_impl(X, y);
    }

    [[nodiscard]] LabelType predict(
        const Eigen::Ref<const MatrixType>& X) const {
        this->check_is_fitted();
        this->validate_feature_count(X);
        return static_cast<const Derived*>(this)->predict_impl(X);
    }

    [[nodiscard]] ScalarType score(
        const Eigen::Ref<const MatrixType>& X,
        const Eigen::Ref<const LabelType>& y) const {
        this->check_is_fitted();
        this->validate_feature_count(X);
        LabelType preds = static_cast<const Derived*>(this)->predict_impl(X);
        int correct = 0;
        for (Eigen::Index i = 0; i < y.size(); ++i) {
            if (preds(i) == y(i)) ++correct;
        }
        return static_cast<ScalarType>(correct) /
               static_cast<ScalarType>(y.size());
    }
};

} // namespace Skigen

#endif // SKIGEN_CORE_BASE_H
