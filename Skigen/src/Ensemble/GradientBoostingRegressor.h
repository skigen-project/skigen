// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_ENSEMBLE_GRADIENT_BOOSTING_REGRESSOR_H
#define SKIGEN_ENSEMBLE_GRADIENT_BOOSTING_REGRESSOR_H

#include "../Core/Base.h"
#include "../Core/Validation.h"
#include "../Tree/DecisionTree.h"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace Skigen {

/// @addtogroup Algo_RandomForest
/// @{

/// @brief Gradient Boosting for regression.
///
/// Stage-wise additive model that fits a sequence of regression trees on the
/// pseudo-residuals of the current ensemble prediction. The final predictor
/// is
/// @f[ \hat{F}_M(x) = F_0 + \eta \sum_{m=1}^{M} h_m(x) @f]
/// where @f$ F_0 = \bar{y} @f$ initialises the ensemble at the marginal mean
/// (sklearn's `init="zero"` default uses a `DummyRegressor` that fits the
/// mean — Skigen mirrors that behaviour exactly), @f$ \eta @f$ is the
/// learning rate, and each @f$ h_m @f$ is a `DecisionTreeRegressor`
/// trained on the residuals of stage @f$ m-1 @f$.
///
/// Mirrors
/// [sklearn.ensemble.GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html).
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default |
/// |---|---|---|
/// | `loss` | `Loss` | `SquaredError` |
/// | `learning_rate` | `Scalar` | `0.1` |
/// | `n_estimators` | `int` | `100` |
/// | `subsample` | `Scalar` | `1.0` |
/// | `criterion` | `CriterionGB` | `FriedmanMSE` |
/// | `min_samples_split` | `int` | `2` |
/// | `min_samples_leaf` | `int` | `1` |
/// | `min_weight_fraction_leaf` | `Scalar` | `0.0` *(deprecated, no-op)* |
/// | `max_depth` | `int` | `3` |
/// | `min_impurity_decrease` | `Scalar` | `0.0` *(deprecated, no-op)* |
/// | `random_state` | `optional<uint64_t>` | `nullopt` |
/// | `alpha` | `Scalar` | `0.9` *(deprecated, no-op for SquaredError)* |
/// | `verbose` | `int` | `0` |
/// | `max_leaf_nodes` | `optional<int>` | `nullopt` |
/// | `warm_start` | `bool` | `false` *(deprecated, no-op)* |
/// | `validation_fraction` | `Scalar` | `0.1` |
/// | `n_iter_no_change` | `optional<int>` | `nullopt` |
/// | `tol` | `Scalar` | `1e-4` |
/// | `ccp_alpha` | `Scalar` | `0.0` *(deprecated, no-op)* |
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type | Description |
/// |---|---|---|
/// | `estimators()` | `vector<DecisionTreeRegressor>` | Fitted base trees. |
/// | `init()` | `Scalar` | Initial prediction (mean of y). |
/// | `n_estimators_fitted()` | `int` | Number of stages actually fitted. |
/// | `feature_importances()` | `RowVectorType` | Mean tree feature importances. |
/// | `train_score()` | `VectorType` | Per-stage training MSE (length n_stages). |
///
/// ### Limitations relative to scikit-learn Only `loss=SquaredError`,
///   `subsample=1.0`, and `criterion=FriedmanMSE` are honoured .
///   `AbsoluteError`, `Huber`, `Quantile`, stochastic boosting,
///   early stopping, `warm_start`, `ccp_alpha`, `min_impurity_decrease`,
///   `max_leaf_nodes`, and `min_weight_fraction_leaf` are accepted as
///   constructor parameters but are not honoured at fit time.
///   `max_features` is forwarded to the underlying tree via the
///   explicit mode/value pair.
template <typename Scalar = double>
class GradientBoostingRegressor
    : public Predictor<GradientBoostingRegressor<Scalar>, Scalar> {
public:
    using Base = Predictor<GradientBoostingRegressor<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;
    using typename Base::ScalarType;

    enum class Loss { SquaredError, AbsoluteError, Huber, Quantile };
    enum class CriterionGB { FriedmanMSE, SquaredError };

    explicit GradientBoostingRegressor(
        Loss loss = Loss::SquaredError,
        Scalar learning_rate = Scalar{0.1},
        int n_estimators = 100,
        Scalar subsample = Scalar{1.0},
        CriterionGB criterion = CriterionGB::FriedmanMSE,
        int min_samples_split = 2,
        int min_samples_leaf = 1,
        Scalar min_weight_fraction_leaf = Scalar{0},
        int max_depth = 3,
        Scalar min_impurity_decrease = Scalar{0},
        std::optional<uint64_t> random_state = std::nullopt,
        Scalar alpha = Scalar{0.9},
        int verbose = 0,
        std::optional<int> max_leaf_nodes = std::nullopt,
        bool warm_start = false,
        Scalar validation_fraction = Scalar{0.1},
        std::optional<int> n_iter_no_change = std::nullopt,
        Scalar tol = Scalar{1e-4},
        Scalar ccp_alpha = Scalar{0})
        : loss_(loss),
          learning_rate_(learning_rate),
          n_estimators_(n_estimators),
          subsample_(subsample),
          criterion_(criterion),
          min_samples_split_(min_samples_split),
          min_samples_leaf_(min_samples_leaf),
          min_weight_fraction_leaf_(min_weight_fraction_leaf),
          max_depth_(max_depth),
          min_impurity_decrease_(min_impurity_decrease),
          random_state_(random_state),
          alpha_(alpha),
          verbose_(verbose),
          max_leaf_nodes_(max_leaf_nodes),
          warm_start_(warm_start),
          validation_fraction_(validation_fraction),
          n_iter_no_change_(n_iter_no_change),
          tol_(tol),
          ccp_alpha_(ccp_alpha) {
        if (subsample_ <= Scalar{0} || subsample_ > Scalar{1}) {
            throw std::invalid_argument(
                "subsample must be in (0, 1]; got " +
                std::to_string(static_cast<double>(subsample_)));
        }
        if (subsample_ < Scalar{1}) {
            throw std::invalid_argument(
                "GradientBoostingRegressor: subsample < 1.0 (stochastic GB) is "
                "not implemented.");
        }
        if ((loss_ == Loss::Quantile || loss_ == Loss::Huber) &&
            (alpha_ <= Scalar{0} || alpha_ >= Scalar{1})) {
            throw std::invalid_argument(
                "GradientBoostingRegressor: alpha must be in (0, 1) for "
                "Quantile and Huber losses.");
        }
    }

    // -- Accessors ----------------------------------------------------------

    [[nodiscard]] Loss loss() const noexcept { return loss_; }
    [[nodiscard]] Scalar learning_rate() const noexcept { return learning_rate_; }
    [[nodiscard]] int n_estimators() const noexcept { return n_estimators_; }
    [[nodiscard]] int max_depth() const noexcept { return max_depth_; }
    [[nodiscard]] Scalar subsample() const noexcept { return subsample_; }

    [[nodiscard]] Scalar init() const {
        this->check_is_fitted(); return init_;
    }
    [[nodiscard]] int n_estimators_fitted() const {
        this->check_is_fitted();
        return static_cast<int>(estimators_storage_.size());
    }
    [[nodiscard]] const std::vector<DecisionTreeRegressor<Scalar>>&
    estimators() const {
        this->check_is_fitted(); return estimators_storage_;
    }
    [[nodiscard]] const RowVectorType& feature_importances() const {
        this->check_is_fitted(); return feature_importances_;
    }
    [[nodiscard]] const VectorType& train_score() const {
        this->check_is_fitted(); return train_score_;
    }

    SKIGEN_PARAMS(
        (learning_rate,            learning_rate_,            double),
        (n_estimators,             n_estimators_,             int),
        (subsample,                subsample_,                double),
        (min_samples_split,        min_samples_split_,        int),
        (min_samples_leaf,         min_samples_leaf_,         int),
        (min_weight_fraction_leaf, min_weight_fraction_leaf_,  double),
        (max_depth,                max_depth_,                int),
        (min_impurity_decrease,    min_impurity_decrease_,     double),
        (alpha,                    alpha_,                    double),
        (verbose,                  verbose_,                  int),
        (warm_start,               warm_start_,               bool),
        (validation_fraction,      validation_fraction_,       double),
        (tol,                      tol_,                      double),
        (ccp_alpha,                ccp_alpha_,                double))

    // -- Fit/Predict --------------------------------------------------------

    GradientBoostingRegressor& fit_impl(
        const Eigen::Ref<const MatrixType>& X,
        const Eigen::Ref<const VectorType>& y) {
        internal::check_non_empty(X);
        internal::check_consistent_length(X, y);

        this->n_features_in_ = X.cols();
        const Eigen::Index n = X.rows();

        // The init estimate is the loss-optimal constant prediction.
        init_ = initial_estimate(y);

        VectorType F = VectorType::Constant(n, init_);
        estimators_storage_.clear();
        estimators_storage_.reserve(static_cast<std::size_t>(n_estimators_));
        scale_steps_.clear();
        scale_steps_.reserve(static_cast<std::size_t>(n_estimators_));

        feature_importances_ = RowVectorType::Zero(X.cols());
        train_score_ = VectorType::Zero(n_estimators_);

        const uint64_t base_seed = random_state_.value_or(0ULL);

        for (int stage = 0; stage < n_estimators_; ++stage) {
            const VectorType residuals = negative_gradient(y, F);

            DecisionTreeRegressor<Scalar> tree(
                max_depth_,
                min_samples_split_,
                /*max_features_mode=*/0,
                /*max_features_value=*/0.0,
                random_state_.has_value()
                    ? std::optional<uint64_t>(base_seed ^
                          static_cast<uint64_t>(stage))
                    : std::nullopt);
            tree.fit(X, residuals);

            VectorType update = tree.predict(X);
            // For non-squared losses, take a single loss-optimal global step so
            // the leaf gradients are scaled to minimise the actual loss along
            // the tree direction (a stable line search, no per-leaf grouping).
            const Scalar step = loss_ == Loss::SquaredError
                ? Scalar{1}
                : line_search_step(y, F, update);
            scale_steps_.push_back(step);
            F.noalias() += learning_rate_ * step * update;

            train_score_(stage) =
                loss_value(y, F) / static_cast<Scalar>(n);

            // Aggregate feature importances when the tree exposes them.
            feature_importances_ += tree.feature_importances();
            estimators_storage_.push_back(std::move(tree));
        }

        if (n_estimators_ > 0) {
            feature_importances_ /= static_cast<Scalar>(n_estimators_);
            const Scalar fi_sum = feature_importances_.sum();
            if (fi_sum > Scalar{0}) feature_importances_ /= fi_sum;
        }

        this->fitted_ = true;
        return *this;
    }

    [[nodiscard]] VectorType predict_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        VectorType F = VectorType::Constant(X.rows(), init_);
        for (std::size_t m = 0; m < estimators_storage_.size(); ++m) {
            F.noalias() += learning_rate_ * scale_steps_[m] *
                           estimators_storage_[m].predict(X);
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
    // Constructor-fixed parameters
    Loss loss_;
    Scalar learning_rate_;
    int n_estimators_;
    Scalar subsample_;
    CriterionGB criterion_;
    int min_samples_split_;
    int min_samples_leaf_;
    Scalar min_weight_fraction_leaf_;
    int max_depth_;
    Scalar min_impurity_decrease_;
    std::optional<uint64_t> random_state_;
    Scalar alpha_;
    int verbose_;
    std::optional<int> max_leaf_nodes_;
    bool warm_start_;
    Scalar validation_fraction_;
    std::optional<int> n_iter_no_change_;
    Scalar tol_;
    Scalar ccp_alpha_;

    // Fitted state
    Scalar init_{0};
    std::vector<DecisionTreeRegressor<Scalar>> estimators_storage_;
    std::vector<Scalar> scale_steps_;
    RowVectorType feature_importances_;
    VectorType train_score_;

    // -- Loss helpers -------------------------------------------------------

    [[nodiscard]] static Scalar percentile(VectorType values, Scalar q) {
        std::sort(values.data(), values.data() + values.size());
        if (values.size() == 1) return values(0);
        const Scalar rank = q * static_cast<Scalar>(values.size() - 1);
        const Eigen::Index lo = static_cast<Eigen::Index>(std::floor(rank));
        const Eigen::Index hi = std::min(lo + 1, values.size() - 1);
        const Scalar frac = rank - static_cast<Scalar>(lo);
        return values(lo) * (Scalar{1} - frac) + values(hi) * frac;
    }

    [[nodiscard]] static Scalar median(VectorType values) {
        return percentile(std::move(values), Scalar{0.5});
    }

    [[nodiscard]] Scalar initial_estimate(const VectorType& y) const {
        switch (loss_) {
            case Loss::SquaredError:
                return y.mean();
            case Loss::AbsoluteError:
            case Loss::Huber:
                return median(y);
            case Loss::Quantile:
                return percentile(y, alpha_);
        }
        return y.mean();
    }

    [[nodiscard]] VectorType negative_gradient(const VectorType& y,
                                               const VectorType& F) const {
        const VectorType residual = y - F;
        switch (loss_) {
            case Loss::SquaredError:
                return residual;
            case Loss::AbsoluteError:
                return residual.array().sign().matrix();
            case Loss::Huber: {
                const Scalar delta = percentile(residual.cwiseAbs(), alpha_);
                VectorType g(residual.size());
                for (Eigen::Index i = 0; i < residual.size(); ++i) {
                    const Scalar r = residual(i);
                    g(i) = std::abs(r) <= delta
                        ? r
                        : delta * (r > Scalar{0} ? Scalar{1} : Scalar{-1});
                }
                return g;
            }
            case Loss::Quantile: {
                VectorType g(residual.size());
                for (Eigen::Index i = 0; i < residual.size(); ++i) {
                    g(i) = residual(i) > Scalar{0} ? alpha_ : alpha_ - Scalar{1};
                }
                return g;
            }
        }
        return residual;
    }

    // Find the scalar step that minimises the loss along the tree direction:
    //   argmin_s  loss(y, F + s * update).
    // Uses a coarse golden-section-style scan over a bounded range; robust for
    // the convex per-stage 1-D objective without per-leaf bookkeeping.
    [[nodiscard]] Scalar line_search_step(const VectorType& y,
                                          const VectorType& F,
                                          const VectorType& update) const {
        const Scalar denom = update.squaredNorm();
        if (!(denom > Scalar{0})) return Scalar{0};
        // Squared-error optimal step is the closed form; use it as the scan
        // centre so the search stays well-scaled for all losses.
        const Scalar centre = (y - F).dot(update) / denom;
        const Scalar hi = std::abs(centre) > Scalar{0}
            ? Scalar{4} * std::abs(centre)
            : Scalar{1};
        Scalar best_s = Scalar{0};
        Scalar best_loss = loss_value(y, F);
        constexpr int steps = 80;
        for (int i = 0; i <= steps; ++i) {
            const Scalar s = -hi + (Scalar{2} * hi) *
                static_cast<Scalar>(i) / static_cast<Scalar>(steps);
            const Scalar l = loss_value(y, (F + s * update).eval());
            if (l < best_loss) {
                best_loss = l;
                best_s = s;
            }
        }
        return best_s;
    }

    [[nodiscard]] Scalar loss_value(const VectorType& y,
                                    const VectorType& F) const {
        const VectorType residual = y - F;
        switch (loss_) {
            case Loss::SquaredError:
                return residual.array().square().sum();
            case Loss::AbsoluteError:
                return residual.array().abs().sum();
            case Loss::Huber: {
                const Scalar delta = percentile(residual.cwiseAbs(), alpha_);
                Scalar total{0};
                for (Eigen::Index i = 0; i < residual.size(); ++i) {
                    const Scalar a = std::abs(residual(i));
                    total += a <= delta
                        ? Scalar{0.5} * residual(i) * residual(i)
                        : delta * (a - Scalar{0.5} * delta);
                }
                return total;
            }
            case Loss::Quantile: {
                Scalar total{0};
                for (Eigen::Index i = 0; i < residual.size(); ++i) {
                    const Scalar r = residual(i);
                    total += r >= Scalar{0} ? alpha_ * r : (alpha_ - Scalar{1}) * r;
                }
                return total;
            }
        }
        return residual.array().square().sum();
    }
};

/// @}

} // namespace Skigen

#endif // SKIGEN_ENSEMBLE_GRADIENT_BOOSTING_REGRESSOR_H
