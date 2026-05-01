// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_PIPELINE_PIPELINE_H
#define SKIGEN_PIPELINE_PIPELINE_H

#include <Eigen/Core>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace Skigen {

/// Pipeline — Compile-time chain of transformers ending with an estimator.
/// Mirrors sklearn.pipeline.Pipeline.
///
/// Usage:
///   auto pipe = Skigen::make_pipeline(
///       Skigen::StandardScaler<double>(),
///       Skigen::LinearRegression<double>());
///   pipe.fit(X, y);
///   auto y_pred = pipe.predict(X);
template <typename... Steps>
class Pipeline {
    static_assert(sizeof...(Steps) >= 1, "Pipeline needs at least one step.");

public:
    using Scalar = double;
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    explicit Pipeline(Steps... steps) : steps_(std::move(steps)...) {}

    /// fit with supervised target (transformers + final estimator)
    template <typename YType>
    Pipeline& fit(const MatrixType& X, const YType& y) {
        MatrixType X_transformed = fit_transform_steps(
            X, std::make_index_sequence<sizeof...(Steps) - 1>{});
        std::get<sizeof...(Steps) - 1>(steps_).fit(X_transformed, y);
        fitted_ = true;
        return *this;
    }

    /// predict (transform through all steps, predict with final)
    template <typename... Args>
    [[nodiscard]] auto predict(const MatrixType& X, Args&&... args) const {
        if (!fitted_) throw std::runtime_error("Pipeline not fitted.");
        MatrixType X_transformed = transform_steps(
            X, std::make_index_sequence<sizeof...(Steps) - 1>{});
        return std::get<sizeof...(Steps) - 1>(steps_).predict(
            X_transformed, std::forward<Args>(args)...);
    }

    /// score (transform through all steps, score with final)
    template <typename YType>
    [[nodiscard]] auto score(const MatrixType& X, const YType& y) const {
        if (!fitted_) throw std::runtime_error("Pipeline not fitted.");
        MatrixType X_transformed = transform_steps(
            X, std::make_index_sequence<sizeof...(Steps) - 1>{});
        return std::get<sizeof...(Steps) - 1>(steps_).score(X_transformed, y);
    }

    /// Access a step by index
    template <std::size_t I>
    [[nodiscard]] auto& get() { return std::get<I>(steps_); }

    template <std::size_t I>
    [[nodiscard]] const auto& get() const { return std::get<I>(steps_); }

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

/// Factory function for creating pipelines with deduced types.
template <typename... Steps>
[[nodiscard]] Pipeline<Steps...> make_pipeline(Steps... steps) {
    return Pipeline<Steps...>(std::move(steps)...);
}

} // namespace Skigen

#endif // SKIGEN_PIPELINE_PIPELINE_H
