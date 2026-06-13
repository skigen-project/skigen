// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_PIPELINE_PIPELINE_H
#define SKIGEN_PIPELINE_PIPELINE_H

#include "../Core/Params.h"

#include <Eigen/Core>
#include <stdexcept>
#include <string>
#include <string_view>
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
/// ### Limitations relative to scikit-learn
///
/// The following scikit-learn features
///   are not honoured: `memory` (caching), `verbose`,
///   named steps (steps are accessed by index), `set_params()`,
///   `get_params()`, `set_output()`.
///   `fit_transform()`, `fit_predict()`, and `inverse_transform()`
///   are not implemented.
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

    /// @brief Route a `"<step_index>__<param>"` setting to the addressed step.
    ///
    /// Enables hyperparameter search over pipeline steps: a `GridSearchCV`
    /// parameter grid keyed by e.g. `"1__alpha"` updates `alpha` on step 1.
    /// The pipeline is an index-addressed tuple of steps, so the step prefix
    /// is the numeric index (sklearn addresses steps by name; Skigen's
    /// compile-time tuple has no names).
    void set_param(std::string_view name, const ParameterValue& value) {
        const auto sep = name.find("__");
        if (sep == std::string_view::npos) {
            throw std::invalid_argument(
                "Pipeline::set_param: expected '<step_index>__<param>', got '" +
                std::string(name) + "'");
        }
        const std::string index_str(name.substr(0, sep));
        std::size_t index = 0;
        try {
            std::size_t consumed = 0;
            index = static_cast<std::size_t>(std::stoul(index_str, &consumed));
            if (consumed != index_str.size()) throw std::invalid_argument("");
        } catch (const std::exception&) {
            throw std::invalid_argument(
                "Pipeline::set_param: step prefix '" + index_str +
                "' is not a numeric index");
        }
        const std::string_view param = name.substr(sep + 2);
        const bool routed = route_set_param(
            index, param, value, std::index_sequence_for<Steps...>{});
        if (!routed) {
            throw std::out_of_range(
                "Pipeline::set_param: step index out of range");
        }
    }

private:
    template <std::size_t I>
    bool route_set_param_single(std::size_t index, std::string_view param,
                                const ParameterValue& value) {
        if (index != I) return false;
        auto& step = std::get<I>(steps_);
        if constexpr (requires { step.set_param(param, value); }) {
            step.set_param(param, value);
        } else {
            throw std::invalid_argument(
                "Pipeline::set_param: step " + std::to_string(I) +
                " does not expose set_param");
        }
        return true;
    }

    template <std::size_t... Is>
    bool route_set_param(std::size_t index, std::string_view param,
                         const ParameterValue& value,
                         std::index_sequence<Is...>) {
        return (route_set_param_single<Is>(index, param, value) || ...);
    }

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
