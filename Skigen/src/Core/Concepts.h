// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_CORE_CONCEPTS_H
#define SKIGEN_CORE_CONCEPTS_H

#include <concepts>
#include <type_traits>

namespace Skigen {

// ---------------------------------------------------------------------------
// Scalar constraint: must be a floating-point type
// ---------------------------------------------------------------------------

template <typename T>
concept EigenScalar = std::is_floating_point_v<T>;

// ---------------------------------------------------------------------------
// Estimator concept: any type with is_fitted()
// ---------------------------------------------------------------------------

template <typename T>
concept EstimatorLike = requires(T t) {
    { t.is_fitted() } -> std::convertible_to<bool>;
};

// ---------------------------------------------------------------------------
// Transformer concept: Estimator + transform / inverse_transform
// ---------------------------------------------------------------------------

template <typename T>
concept TransformerLike = EstimatorLike<T> &&
    requires(T t, const typename T::MatrixType& X) {
        { t.fit(X) } -> std::same_as<T&>;
        { t.transform(X) } -> std::same_as<typename T::MatrixType>;
        { t.fit_transform(X) } -> std::same_as<typename T::MatrixType>;
    };

// ---------------------------------------------------------------------------
// Predictor concept: Estimator + predict / score
// ---------------------------------------------------------------------------

template <typename T>
concept PredictorLike = EstimatorLike<T> &&
    requires(T t,
             const typename T::MatrixType& X,
             const typename T::VectorType& y) {
        { t.fit(X, y) } -> std::same_as<T&>;
        { t.predict(X) } -> std::same_as<typename T::VectorType>;
        { t.score(X, y) } -> std::same_as<typename T::ScalarType>;
    };

} // namespace Skigen

#endif // SKIGEN_CORE_CONCEPTS_H
