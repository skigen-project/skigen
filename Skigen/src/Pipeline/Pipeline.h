// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_PIPELINE_PIPELINE_H
#define SKIGEN_PIPELINE_PIPELINE_H

#include <Eigen/Core>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace Skigen {

/// @defgroup Algo_Pipeline Pipeline
/// @ingroup Pipeline
/// @brief Compile-time chain of transformers ending with an estimator.
/// @{

/// @brief Pipeline of transforms with a final estimator.
///
/// Sequentially apply a list of transforms and a final estimator.
/// Intermediate steps of the pipeline must implement `fit()` and
/// `transform()`. The final estimator only needs to implement `fit()`.
///
/// The pipeline object is created via `make_pipeline()`.
///
/// Mirrors
/// [sklearn.pipeline.Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html).
///
/// ### Notes
///
/// This is a compile-time pipeline using variadic templates and
/// `std::tuple`. The number and types of steps are fixed at compile
/// time, which enables full inlining and zero runtime overhead.
///
/// @note **scikit-learn parity gaps:** The following sklearn features
///   are not yet supported: `memory` (caching), `verbose`,
///   named steps (steps are accessed by index), `set_params()`,
///   `get_params()`, `set_output()`.
///   `fit_transform()`, `fit_predict()`, and `inverse_transform()`
///   are not yet implemented.
///
/// ### Examples
///
/// @snippet pipeline.cpp example_pipeline
///
/// **Usage:**
/// ```cpp
/// auto pipe = Skigen::make_pipeline(
///     Skigen::StandardScaler<double>(),
///     Skigen::LinearRegression<double>());
/// pipe.fit(X, y);
/// auto y_pred = pipe.predict(X);
/// ```
template <typename... Steps>
class Pipeline {
    static_assert(sizeof...(Steps) >= 1, "Pipeline needs at least one step.");

public:
    using Scalar = double;
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    /// @brief Construct a Pipeline from individual steps.
    /// @param steps The transformer and estimator steps (moved into the pipeline).
    explicit Pipeline(Steps... steps) : steps_(std::move(steps)...) {}

    /// @brief Fit all transformers and the final estimator.
    ///
    /// Calls `fit()` then `transform()` on each intermediate step,
    /// then `fit()` on the final estimator with the transformed data.
    ///
    /// @param X Training data of shape (n_samples, n_features).
    /// @param y Target values (type depends on the final estimator).
    /// @return Reference to the fitted pipeline (`*this`).
    /// fit with supervised target (transformers + final estimator)
    template <typename YType>
    Pipeline& fit(const MatrixType& X, const YType& y) {
        MatrixType X_transformed = fit_transform_steps(
            X, std::make_index_sequence<sizeof...(Steps) - 1>{});
        std::get<sizeof...(Steps) - 1>(steps_).fit(X_transformed, y);
        fitted_ = true;
        return *this;
    }

    /// @brief Predict using the pipeline.
    ///
    /// Applies `transform()` on all intermediate steps, then calls
    /// `predict()` on the final estimator.
    ///
    /// @param X Test data of shape (n_samples, n_features).
    /// @return Predictions from the final estimator.
    /// @throws std::runtime_error if the pipeline has not been fitted.
    /// predict (transform through all steps, predict with final)
    template <typename... Args>
    [[nodiscard]] auto predict(const MatrixType& X, Args&&... args) const {
        if (!fitted_) throw std::runtime_error("Pipeline not fitted.");
        MatrixType X_transformed = transform_steps(
            X, std::make_index_sequence<sizeof...(Steps) - 1>{});
        return std::get<sizeof...(Steps) - 1>(steps_).predict(
            X_transformed, std::forward<Args>(args)...);
    }

    /// @brief Score the pipeline on test data.
    ///
    /// Applies `transform()` on all intermediate steps, then calls
    /// `score()` on the final estimator.
    ///
    /// @param X Test data of shape (n_samples, n_features).
    /// @param y True target values.
    /// @return Score from the final estimator.
    /// @throws std::runtime_error if the pipeline has not been fitted.
    /// score (transform through all steps, score with final)
    template <typename YType>
    [[nodiscard]] auto score(const MatrixType& X, const YType& y) const {
        if (!fitted_) throw std::runtime_error("Pipeline not fitted.");
        MatrixType X_transformed = transform_steps(
            X, std::make_index_sequence<sizeof...(Steps) - 1>{});
        return std::get<sizeof...(Steps) - 1>(steps_).score(X_transformed, y);
    }

    /// @brief Access a step by compile-time index.
    ///
    /// @tparam I Zero-based index of the step.
    /// @return Reference to the step.
    template <std::size_t I>
    [[nodiscard]] auto& get() { return std::get<I>(steps_); }

    /// @brief Access a step by compile-time index (const).
    /// @tparam I Zero-based index of the step.
    /// @return Const reference to the step.
    template <std::size_t I>
    [[nodiscard]] const auto& get() const { return std::get<I>(steps_); }

    /// @brief Check whether the pipeline has been fitted.
    /// @return `true` if `fit()` has been called successfully.
    [[nodiscard]] bool is_fitted() const noexcept { return fitted_; }

private:
    std::tuple<Steps...> steps_;
    bool fitted_ = false;

    // fit_transform through transformer steps [0, N-1)
    template <std::size_t... Is>
    MatrixType fit_transform_steps(const MatrixType& X,
                                   std::index_sequence<Is...>) {
        MatrixType result = X;
        ((result = fit_transform_single<Is>(result)), ...);
        return result;
    }

    template <std::size_t I>
    MatrixType fit_transform_single(const MatrixType& X) {
        auto& step = std::get<I>(steps_);
        step.fit(X);
        return step.transform(X);
    }

    // transform through transformer steps [0, N-1)
    template <std::size_t... Is>
    MatrixType transform_steps(const MatrixType& X,
                               std::index_sequence<Is...>) const {
        MatrixType result = X;
        ((result = std::get<Is>(steps_).transform(result)), ...);
        return result;
    }
};

/// @brief Factory function for creating pipelines with deduced types.
///
/// @param steps The transformer and estimator steps.
/// @return A Pipeline composed of the given steps.
///
/// **Usage:**
/// ```cpp
/// auto pipe = Skigen::make_pipeline(
///     Skigen::StandardScaler<double>(),
///     Skigen::Ridge<double>(0.5));
/// ```
template <typename... Steps>
[[nodiscard]] Pipeline<Steps...> make_pipeline(Steps... steps) {
    return Pipeline<Steps...>(std::move(steps)...);
}

/// @}

} // namespace Skigen

#endif // SKIGEN_PIPELINE_PIPELINE_H
