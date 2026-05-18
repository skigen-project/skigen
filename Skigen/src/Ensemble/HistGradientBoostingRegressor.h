// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_ENSEMBLE_HIST_GRADIENT_BOOSTING_REGRESSOR_H
#define SKIGEN_ENSEMBLE_HIST_GRADIENT_BOOSTING_REGRESSOR_H

#include "../Core/Base.h"
#include "../Core/Validation.h"
#include "../Tree/DecisionTree.h"

#include <Eigen/Core>
#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace Skigen {

/// @addtogroup Algo_RandomForest
/// @{

/// @brief Histogram-based Gradient Boosting for regression.
///
/// Bins each feature into at most `max_bins` quantile-based buckets up
/// front, then runs stage-wise additive gradient boosting on the binned
/// representation. The binning step is what gives sklearn's
/// `HistGradientBoostingRegressor` its scaling advantage on large
/// datasets — split candidates collapse from `n_samples` distinct
/// thresholds per feature down to `max_bins`.
///
/// Mirrors
/// [sklearn.ensemble.HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)
/// for the squared-error loss case.
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default |
/// |---|---|---|
/// | `loss` | `Loss` | `SquaredError` |
/// | `learning_rate` | `Scalar` | `0.1` |
/// | `max_iter` | `int` | `100` |
/// | `max_leaf_nodes` | `optional<int>` | `31` *(accepted but not enforced — depth-first growth is used instead of leaf-wise)* |
/// | `max_depth` | `optional<int>` | `nullopt` *(unlimited)* |
/// | `min_samples_leaf` | `int` | `20` |
/// | `l2_regularization` | `Scalar` | `0.0` *(accepted but currently ignored)* |
/// | `max_bins` | `int` | `255` |
/// | `categorical_features` | `optional<vector<int>>` | `nullopt` *(accepted but ignored — categoricals treated as ordinals)* |
/// | `monotonic_cst` | `optional<vector<int>>` | `nullopt` *(accepted but ignored)* |
/// | `early_stopping` | `bool` | `false` *(accepted but ignored — full max_iter is always run)* |
/// | `tol` | `Scalar` | `1e-7` *(accepted but ignored)* |
/// | `random_state` | `optional<uint64_t>` | `nullopt` |
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type |
/// |---|---|
/// | `init()` | `Scalar` (mean of y) |
/// | `n_iter()` | `int` (== max_iter; no early-stopping yet) |
/// | `bin_edges()` | `vector<vector<Scalar>>` (per-feature thresholds) |
/// | `train_score()` | `VectorType` (per-stage MSE) |
///
/// ### Limitations relative to scikit-learn
///
/// Only `loss=SquaredError` is supported; AbsoluteError, Poisson, and
/// Quantile losses raise on construction. Once X is binned, the
/// existing `DecisionTreeRegressor` is used for split selection
/// (sort-based, not the native histogram-based split finder) — this
/// produces equivalent predictions when the binning is fine enough but
/// does **not** realise the scaling-on-large-n advantage that a
/// histogram split finder would. Leaf-wise growth (`max_leaf_nodes`),
/// native categoricals, monotonic constraints, early stopping, and
/// `l2_regularization` are accepted as constructor parameters but are
/// not honoured at fit time.
template <typename Scalar = double>
class HistGradientBoostingRegressor
    : public Predictor<HistGradientBoostingRegressor<Scalar>, Scalar> {
public:
    using Base = Predictor<HistGradientBoostingRegressor<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;
    using typename Base::ScalarType;

    enum class Loss { SquaredError, AbsoluteError, Poisson, Quantile };

    explicit HistGradientBoostingRegressor(
        Loss loss = Loss::SquaredError,
        Scalar learning_rate = Scalar{0.1},
        int max_iter = 100,
        std::optional<int> max_leaf_nodes = 31,
        std::optional<int> max_depth = std::nullopt,
        int min_samples_leaf = 20,
        Scalar l2_regularization = Scalar{0},
        int max_bins = 255,
        std::optional<std::vector<int>> categorical_features = std::nullopt,
        std::optional<std::vector<int>> monotonic_cst = std::nullopt,
        bool early_stopping = false,
        Scalar tol = Scalar{1e-7},
        std::optional<uint64_t> random_state = std::nullopt)
        : loss_(loss),
          learning_rate_(learning_rate),
          max_iter_(max_iter),
          max_leaf_nodes_(max_leaf_nodes),
          max_depth_(max_depth),
          min_samples_leaf_(min_samples_leaf),
          l2_regularization_(l2_regularization),
          max_bins_(max_bins),
          categorical_features_(std::move(categorical_features)),
          monotonic_cst_(std::move(monotonic_cst)),
          early_stopping_(early_stopping),
          tol_(tol),
          random_state_(random_state) {
        if (loss_ != Loss::SquaredError) {
            throw std::invalid_argument(
                "HistGradientBoostingRegressor: only loss=SquaredError is "
                "implemented.");
        }
        if (max_bins_ < 2 || max_bins_ > 255) {
            throw std::invalid_argument(
                "max_bins must be in [2, 255]; got " +
                std::to_string(max_bins_));
        }
    }

    // -- Accessors ----------------------------------------------------------

    [[nodiscard]] Loss   loss()          const noexcept { return loss_; }
    [[nodiscard]] Scalar learning_rate() const noexcept { return learning_rate_; }
    [[nodiscard]] int    max_iter()      const noexcept { return max_iter_; }
    [[nodiscard]] int    max_bins()      const noexcept { return max_bins_; }

    [[nodiscard]] Scalar init() const {
        this->check_is_fitted(); return init_;
    }
    [[nodiscard]] int n_iter() const {
        this->check_is_fitted();
        return static_cast<int>(estimators_.size());
    }
    [[nodiscard]] const std::vector<std::vector<Scalar>>& bin_edges() const {
        this->check_is_fitted(); return bin_edges_;
    }
    [[nodiscard]] const VectorType& train_score() const {
        this->check_is_fitted(); return train_score_;
    }

    SKIGEN_PARAMS(
        (learning_rate,      learning_rate_,      double),
        (max_iter,           max_iter_,           int),
        (min_samples_leaf,   min_samples_leaf_,   int),
        (l2_regularization,  l2_regularization_,  double),
        (max_bins,           max_bins_,           int),
        (early_stopping,     early_stopping_,     bool),
        (tol,                tol_,                double))

    // -- Fit / Predict ------------------------------------------------------

    HistGradientBoostingRegressor& fit_impl(
        const Eigen::Ref<const MatrixType>& X,
        const Eigen::Ref<const VectorType>& y) {
        internal::check_non_empty(X);
        internal::check_consistent_length(X, y);
        this->n_features_in_ = X.cols();
        const Eigen::Index n = X.rows();
        const Eigen::Index p = X.cols();

        // 1. Bin each feature using quantile-based thresholds.
        bin_edges_.assign(static_cast<std::size_t>(p), {});
        MatrixType X_binned(n, p);
        for (Eigen::Index j = 0; j < p; ++j) {
            std::vector<Scalar> col(static_cast<std::size_t>(n));
            for (Eigen::Index i = 0; i < n; ++i) col[i] = X(i, j);
            std::vector<Scalar> sorted_col = col;
            std::sort(sorted_col.begin(), sorted_col.end());

            std::vector<Scalar> thresholds;
            const int n_thresh = max_bins_ - 1;
            // Quantile-based bin edges.
            for (int b = 1; b <= n_thresh; ++b) {
                const std::size_t idx = std::min(
                    static_cast<std::size_t>(
                        static_cast<double>(b) * static_cast<double>(n) /
                        static_cast<double>(max_bins_)),
                    sorted_col.size() - 1);
                Scalar t = sorted_col[idx];
                if (thresholds.empty() || t > thresholds.back()) {
                    thresholds.push_back(t);
                }
            }
            bin_edges_[static_cast<std::size_t>(j)] = thresholds;

            // Map values to bin indices [0, n_bins - 1].
            for (Eigen::Index i = 0; i < n; ++i) {
                auto it = std::upper_bound(
                    thresholds.begin(), thresholds.end(), X(i, j));
                X_binned(i, j) = static_cast<Scalar>(
                    std::distance(thresholds.begin(), it));
            }
        }

        // 2. Initialise predictions at the marginal mean (matches
        //    sklearn's DummyRegressor(strategy="mean") default init).
        init_ = y.mean();
        VectorType F = VectorType::Constant(n, init_);

        estimators_.clear();
        estimators_.reserve(static_cast<std::size_t>(max_iter_));
        train_score_ = VectorType::Zero(max_iter_);

        const uint64_t base_seed = random_state_.value_or(0ULL);
        const int max_depth_eff = max_depth_.value_or(-1);

        for (int stage = 0; stage < max_iter_; ++stage) {
            VectorType residuals = y - F;

            DecisionTreeRegressor<Scalar> tree(
                max_depth_eff,
                min_samples_leaf_ * 2,             // min_samples_split
                /*max_features_mode=*/0,
                /*max_features_value=*/0.0,
                random_state_.has_value()
                    ? std::optional<uint64_t>(
                          base_seed ^ static_cast<uint64_t>(stage))
                    : std::nullopt);
            tree.fit(X_binned, residuals);

            VectorType update = tree.predict(X_binned);
            F.noalias() += learning_rate_ * update;
            train_score_(stage) =
                ((y - F).array().square().sum()) / static_cast<Scalar>(n);
            estimators_.push_back(std::move(tree));
        }

        this->fitted_ = true;
        return *this;
    }

    [[nodiscard]] VectorType predict_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        const Eigen::Index n = X.rows();
        const Eigen::Index p = X.cols();
        // Bin X using stored thresholds.
        MatrixType X_binned(n, p);
        for (Eigen::Index j = 0; j < p; ++j) {
            const auto& thresholds =
                bin_edges_[static_cast<std::size_t>(j)];
            for (Eigen::Index i = 0; i < n; ++i) {
                auto it = std::upper_bound(
                    thresholds.begin(), thresholds.end(), X(i, j));
                X_binned(i, j) = static_cast<Scalar>(
                    std::distance(thresholds.begin(), it));
            }
        }

        VectorType F = VectorType::Constant(n, init_);
        for (const auto& tree : estimators_) {
            F.noalias() += learning_rate_ * tree.predict(X_binned);
        }
        return F;
    }

    [[nodiscard]] Scalar score_impl(
        const Eigen::Ref<const MatrixType>& X,
        const Eigen::Ref<const VectorType>& y) const {
        VectorType yhat = predict_impl(X);
        const Scalar ym = y.mean();
        Scalar ss_res{0}, ss_tot{0};
        for (Eigen::Index i = 0; i < y.size(); ++i) {
            const Scalar r = y(i) - yhat(i);
            ss_res += r * r;
            const Scalar d = y(i) - ym;
            ss_tot += d * d;
        }
        if (ss_tot == Scalar{0}) return Scalar{0};
        return Scalar{1} - ss_res / ss_tot;
    }

private:
    Loss loss_;
    Scalar learning_rate_;
    int max_iter_;
    std::optional<int> max_leaf_nodes_;
    std::optional<int> max_depth_;
    int min_samples_leaf_;
    Scalar l2_regularization_;
    int max_bins_;
    std::optional<std::vector<int>> categorical_features_;
    std::optional<std::vector<int>> monotonic_cst_;
    bool early_stopping_;
    Scalar tol_;
    std::optional<uint64_t> random_state_;

    Scalar init_{0};
    std::vector<std::vector<Scalar>> bin_edges_;        // per-feature
    std::vector<DecisionTreeRegressor<Scalar>> estimators_;
    VectorType train_score_;
};

/// @}

} // namespace Skigen

#endif // SKIGEN_ENSEMBLE_HIST_GRADIENT_BOOSTING_REGRESSOR_H
