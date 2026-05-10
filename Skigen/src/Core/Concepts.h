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

// ---------------------------------------------------------------------------
// Classifier concept: Estimator + fit(X,y_int)/predict/score with int labels
// ---------------------------------------------------------------------------

template <typename T>
concept ClassifierLike = EstimatorLike<T> &&
    requires(T t,
             const typename T::MatrixType& X,
             const typename T::LabelType& y) {
        { t.fit(X, y) } -> std::same_as<T&>;
        { t.predict(X) } -> std::same_as<typename T::LabelType>;
        { t.score(X, y) } -> std::same_as<typename T::ScalarType>;
    };

// ---------------------------------------------------------------------------
// IncrementalLike concept (v1.1.0 §3.1) — `partial_fit` for streaming /
// online learning. The contract is documented in v1.1.0-requirements §3.1.
// ---------------------------------------------------------------------------

template <typename T>
concept IncrementalUnsupervised = EstimatorLike<T> &&
    requires(T t, const typename T::MatrixType& X) {
        { t.partial_fit(X) } -> std::same_as<T&>;
    };

template <typename T>
concept IncrementalSupervised = EstimatorLike<T> &&
    requires(T t,
             const typename T::MatrixType& X,
             const typename T::VectorType& y) {
        { t.partial_fit(X, y) } -> std::same_as<T&>;
    };

template <typename T>
concept IncrementalLike =
    IncrementalUnsupervised<T> || IncrementalSupervised<T>;

// ---------------------------------------------------------------------------
// MultiOutputRegressor concept (v1.1.0 §3.3) — additive multi-target
// regression API: fit_multi(X, Y) + predict_multi(X) + n_targets().
// ---------------------------------------------------------------------------

template <typename T>
concept MultiOutputRegressorLike = EstimatorLike<T> &&
    requires(T t,
             const typename T::MatrixType& X,
             const typename T::MatrixType& Y) {
        { t.fit_multi(X, Y) } -> std::same_as<T&>;
        { t.predict_multi(X) } -> std::same_as<typename T::MatrixType>;
        { t.n_targets() } -> std::convertible_to<int>;
    };

} // namespace Skigen

#endif // SKIGEN_CORE_CONCEPTS_H
