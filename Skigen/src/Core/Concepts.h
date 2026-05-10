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
// Online-learning concepts: estimators that accept new batches of data via
// `partial_fit` and update their state in place rather than refitting from
// scratch. Required contract: subsequent calls produce fitted attributes
// equivalent (within FP tolerance) to a single `fit` over the concatenated
// batches.
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
// MultiOutputRegressor concept — regressors that expose a multi-target
// API: `fit_multi(X, Y)` accepts a (n_samples × n_targets) response matrix
// and `predict_multi(X)` returns the matching prediction matrix.
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

// ---------------------------------------------------------------------------
// ParametrizedLike — estimators that have registered their hyperparameters
// with the SKIGEN_PARAMS(...) reflection layer, exposing the
// `set_param` / `get_params` API used by hyperparameter-search drivers.
// ---------------------------------------------------------------------------

template <typename T>
concept ParametrizedLike = EstimatorLike<T> &&
    requires(T t, const T ct) {
        { ct.get_params_impl() };
    };

} // namespace Skigen

#endif // SKIGEN_CORE_CONCEPTS_H
