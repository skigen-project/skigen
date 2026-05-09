// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_ISOTONIC_ISOTONIC_REGRESSION_H
#define SKIGEN_ISOTONIC_ISOTONIC_REGRESSION_H

#include "../Core/Base.h"
#include "../Core/Validation.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace Skigen {

/// @defgroup Algo_IsotonicRegression Isotonic Regression
/// @ingroup Isotonic
/// @brief Univariate monotonic regression via the Pool-Adjacent-Violators
///   algorithm (PAVA).
/// @{

/// @brief Out-of-bounds policy for `IsotonicRegression::predict`.
enum class OutOfBounds {
    Nan,    ///< Return NaN for inputs outside the training range.
    Clip,   ///< Clip inputs to the training range before interpolation.
    Raise   ///< Throw `std::invalid_argument` if any input is out of range.
};

/// @brief Direction of the fitted isotonic function.
enum class IsotonicIncreasing {
    True,   ///< Fit a non-decreasing function.
    False,  ///< Fit a non-increasing function.
    Auto    ///< Pick direction by sign of the Spearman correlation of (X, y).
};

/// @brief Univariate monotonic regression with the Pool-Adjacent-Violators
///   algorithm.
///
/// Mirrors
/// [sklearn.isotonic.IsotonicRegression](https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html).
///
/// Solves the weighted constrained least-squares problem
///
/// @f[
///   \min_{\hat{y}} \; \sum_i w_i (y_i - \hat{y}_i)^2
///   \quad \text{subject to} \quad
///   \hat{y}_i \le \hat{y}_{i+1} \;\;\forall i
/// @f]
///
/// (the constraint reverses for `increasing = False`). The fitted function is
/// piecewise-linear; predictions interpolate between adjacent thresholds.
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `y_min` | `std::optional<Scalar>` | `nullopt` | Lower bound on the fitted values. |
/// | `y_max` | `std::optional<Scalar>` | `nullopt` | Upper bound on the fitted values. |
/// | `increasing` | `IsotonicIncreasing` | `True` | Direction of the constraint. |
/// | `out_of_bounds` | `OutOfBounds` | `Nan` | Behaviour when `predict` sees inputs outside the training range. |
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type | Description |
/// |----------|------|-------------|
/// | `X_min()` | `Scalar` | Smallest X seen during fit. |
/// | `X_max()` | `Scalar` | Largest X seen during fit. |
/// | `X_thresholds()` | `VectorType` | Unique X values of the step function. |
/// | `y_thresholds()` | `VectorType` | Fitted values at `X_thresholds_`. |
/// | `increasing_()` | `bool` | Effective monotonicity direction (after `Auto` resolution). |
///
/// @note **sklearn parity gaps:** `feature_names_in_` is not exposed.
///   Spearman-based `Auto` direction is computed here from the sign of the
///   ordinary Pearson correlation of (X, y) — matching sklearn 1.7's
///   internal `check_increasing` helper.
template <typename Scalar = double>
class IsotonicRegression
    : public Predictor<IsotonicRegression<Scalar>, Scalar> {
public:
    using Base = Predictor<IsotonicRegression<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;
    using typename Base::ScalarType;

    /// @brief Construct an isotonic regression estimator.
    explicit IsotonicRegression(
        std::optional<Scalar> y_min = std::nullopt,
        std::optional<Scalar> y_max = std::nullopt,
        IsotonicIncreasing increasing = IsotonicIncreasing::True,
        OutOfBounds out_of_bounds = OutOfBounds::Nan)
        : y_min_(y_min),
          y_max_(y_max),
          increasing_param_(increasing),
          out_of_bounds_(out_of_bounds) {}

    // -- Accessors ----------------------------------------------------------

    [[nodiscard]] std::optional<Scalar> y_min() const noexcept { return y_min_; }
    [[nodiscard]] std::optional<Scalar> y_max() const noexcept { return y_max_; }
    [[nodiscard]] IsotonicIncreasing increasing_param() const noexcept {
        return increasing_param_;
    }
    [[nodiscard]] OutOfBounds out_of_bounds() const noexcept {
        return out_of_bounds_;
    }

    [[nodiscard]] Scalar X_min() const { this->check_is_fitted(); return X_min_; }
    [[nodiscard]] Scalar X_max() const { this->check_is_fitted(); return X_max_; }
    [[nodiscard]] const VectorType& X_thresholds() const {
        this->check_is_fitted(); return X_thresholds_;
    }
    [[nodiscard]] const VectorType& y_thresholds() const {
        this->check_is_fitted(); return y_thresholds_;
    }
    /// @brief Effective monotonicity direction after fitting.
    [[nodiscard]] bool increasing_resolved() const {
        this->check_is_fitted(); return increasing_resolved_;
    }

    // -- Fit ---------------------------------------------------------------

    /// @brief Fit the isotonic regression on a 1-D feature.
    ///
    /// X must have shape (n_samples, 1). For multi-column X this method
    /// throws `std::invalid_argument` (matching sklearn).
    IsotonicRegression& fit_impl(const Eigen::Ref<const MatrixType>& X,
                                 const Eigen::Ref<const VectorType>& y) {
        internal::check_non_empty(X);
        internal::check_consistent_length(X, y);

        if (X.cols() != 1) {
            throw std::invalid_argument(
                "IsotonicRegression expects X with a single feature column; "
                "got " + std::to_string(X.cols()) + " columns.");
        }
        this->n_features_in_ = 1;

        const IndexType n = X.rows();

        // Resolve increasing direction
        bool increasing = true;
        switch (increasing_param_) {
            case IsotonicIncreasing::True:  increasing = true;  break;
            case IsotonicIncreasing::False: increasing = false; break;
            case IsotonicIncreasing::Auto: {
                // Sign of Pearson correlation of (X, y); ties -> increasing.
                const Scalar xm = X.col(0).mean();
                const Scalar ym = y.mean();
                Scalar num = Scalar{0};
                for (IndexType i = 0; i < n; ++i) {
                    num += (X(i, 0) - xm) * (y(i) - ym);
                }
                increasing = !(num < Scalar{0});
                break;
            }
        }
        increasing_resolved_ = increasing;

        // Sort indices by X (stable for determinism with ties).
        std::vector<IndexType> order(static_cast<std::size_t>(n));
        std::iota(order.begin(), order.end(), IndexType{0});
        std::stable_sort(order.begin(), order.end(),
            [&](IndexType a, IndexType b) { return X(a, 0) < X(b, 0); });

        // Build sorted (x, y, w) — ties on X are merged before PAVA so that
        // PAVA produces a single fitted value per unique X (matching sklearn).
        std::vector<Scalar> xs;
        std::vector<Scalar> sum_wy;
        std::vector<Scalar> sum_w;
        xs.reserve(static_cast<std::size_t>(n));
        sum_wy.reserve(static_cast<std::size_t>(n));
        sum_w.reserve(static_cast<std::size_t>(n));

        for (IndexType k = 0; k < n; ++k) {
            const IndexType i = order[static_cast<std::size_t>(k)];
            Scalar yi = y(i);
            if (!increasing) yi = -yi;
            if (!xs.empty() && xs.back() == X(i, 0)) {
                sum_wy.back() += yi;
                sum_w.back()  += Scalar{1};
            } else {
                xs.push_back(X(i, 0));
                sum_wy.push_back(yi);
                sum_w.push_back(Scalar{1});
            }
        }

        // Stack-based PAVA on (sum_wy, sum_w) keyed by mean = sum_wy/sum_w.
        // Stores indices into xs so we can reconstruct thresholds afterwards.
        std::vector<std::size_t> stack_end;       // last xs-index in block
        std::vector<Scalar>      stack_sum_wy;
        std::vector<Scalar>      stack_sum_w;

        for (std::size_t k = 0; k < xs.size(); ++k) {
            std::size_t end_idx = k;
            Scalar wy = sum_wy[k];
            Scalar w  = sum_w[k];
            while (!stack_end.empty()) {
                const Scalar prev_mean = stack_sum_wy.back() / stack_sum_w.back();
                const Scalar cur_mean  = wy / w;
                if (prev_mean > cur_mean) {
                    wy += stack_sum_wy.back();
                    w  += stack_sum_w.back();
                    stack_end.pop_back();
                    stack_sum_wy.pop_back();
                    stack_sum_w.pop_back();
                } else {
                    break;
                }
            }
            stack_end.push_back(end_idx);
            stack_sum_wy.push_back(wy);
            stack_sum_w.push_back(w);
        }

        // Convert stack into a per-x fitted-value array (length xs.size()).
        std::vector<Scalar> fitted(xs.size());
        std::size_t start = 0;
        for (std::size_t b = 0; b < stack_end.size(); ++b) {
            const Scalar mean = stack_sum_wy[b] / stack_sum_w[b];
            for (std::size_t j = start; j <= stack_end[b]; ++j) {
                fitted[j] = mean;
            }
            start = stack_end[b] + 1;
        }

        // Flip back if we negated for decreasing direction.
        if (!increasing) {
            for (auto& v : fitted) v = -v;
        }

        // Apply y_min / y_max bounds.
        if (y_min_) {
            for (auto& v : fitted) v = std::max(v, *y_min_);
        }
        if (y_max_) {
            for (auto& v : fitted) v = std::min(v, *y_max_);
        }

        // Compress duplicate-y consecutive thresholds — sklearn keeps only
        // turning points so the linear interpolation is faithful.
        std::vector<Scalar> Xs_keep;
        std::vector<Scalar> Ys_keep;
        Xs_keep.reserve(xs.size());
        Ys_keep.reserve(xs.size());
        for (std::size_t i = 0; i < xs.size(); ++i) {
            const bool is_endpoint = (i == 0 || i + 1 == xs.size());
            if (is_endpoint) {
                Xs_keep.push_back(xs[i]);
                Ys_keep.push_back(fitted[i]);
                continue;
            }
            // Keep points where the slope changes.
            if (fitted[i] != fitted[i - 1] || fitted[i] != fitted[i + 1]) {
                Xs_keep.push_back(xs[i]);
                Ys_keep.push_back(fitted[i]);
            }
        }

        X_thresholds_.resize(static_cast<IndexType>(Xs_keep.size()));
        y_thresholds_.resize(static_cast<IndexType>(Ys_keep.size()));
        for (std::size_t i = 0; i < Xs_keep.size(); ++i) {
            X_thresholds_(static_cast<IndexType>(i)) = Xs_keep[i];
            y_thresholds_(static_cast<IndexType>(i)) = Ys_keep[i];
        }

        X_min_ = xs.front();
        X_max_ = xs.back();
        this->fitted_ = true;
        return *this;
    }

    /// @brief Predict using the fitted piecewise-linear isotonic function.
    [[nodiscard]] VectorType predict_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        VectorType out(X.rows());
        for (IndexType i = 0; i < X.rows(); ++i) {
            out(i) = interpolate_(X(i, 0));
        }
        return out;
    }

    /// @brief Convenience: same as `predict` for sklearn API parity.
    [[nodiscard]] VectorType transform(
        const Eigen::Ref<const MatrixType>& X) const {
        this->check_is_fitted();
        if (X.cols() != 1) {
            throw std::invalid_argument(
                "IsotonicRegression.transform expects X with a single feature "
                "column; got " + std::to_string(X.cols()) + ".");
        }
        return predict_impl(X);
    }

    /// @brief R^2 score on (X, y).
    [[nodiscard]] Scalar score_impl(
        const Eigen::Ref<const MatrixType>& X,
        const Eigen::Ref<const VectorType>& y) const {
        VectorType yhat = predict_impl(X);
        const Scalar y_mean = y.mean();
        Scalar ss_res = Scalar{0}, ss_tot = Scalar{0};
        for (IndexType i = 0; i < y.size(); ++i) {
            const Scalar r = y(i) - yhat(i);
            ss_res += r * r;
            const Scalar d = y(i) - y_mean;
            ss_tot += d * d;
        }
        if (ss_tot == Scalar{0}) return Scalar{0};
        return Scalar{1} - ss_res / ss_tot;
    }

private:
    Scalar interpolate_(Scalar x) const {
        const auto n = X_thresholds_.size();
        if (n == 0) return std::numeric_limits<Scalar>::quiet_NaN();
        if (n == 1) return y_thresholds_(0);

        if (x < X_min_ || x > X_max_) {
            switch (out_of_bounds_) {
                case OutOfBounds::Raise:
                    throw std::invalid_argument(
                        "IsotonicRegression: input out of training range and "
                        "out_of_bounds = Raise.");
                case OutOfBounds::Nan:
                    return std::numeric_limits<Scalar>::quiet_NaN();
                case OutOfBounds::Clip:
                    x = std::clamp(x, X_min_, X_max_);
                    break;
            }
        }

        // Binary search for the bracketing pair.
        IndexType lo = 0, hi = n - 1;
        while (hi - lo > 1) {
            const IndexType mid = (lo + hi) / 2;
            if (X_thresholds_(mid) <= x) lo = mid; else hi = mid;
        }
        const Scalar xl = X_thresholds_(lo);
        const Scalar xh = X_thresholds_(hi);
        if (xh == xl) return y_thresholds_(lo);
        const Scalar t = (x - xl) / (xh - xl);
        return y_thresholds_(lo) + t * (y_thresholds_(hi) - y_thresholds_(lo));
    }

    // Constructor params
    std::optional<Scalar>  y_min_;
    std::optional<Scalar>  y_max_;
    IsotonicIncreasing     increasing_param_;
    OutOfBounds            out_of_bounds_;

    // Fitted state
    Scalar      X_min_ = Scalar{0};
    Scalar      X_max_ = Scalar{0};
    VectorType  X_thresholds_;
    VectorType  y_thresholds_;
    bool        increasing_resolved_ = true;
};

/// @}

} // namespace Skigen

#endif // SKIGEN_ISOTONIC_ISOTONIC_REGRESSION_H
