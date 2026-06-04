// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_FEATURE_SELECTION_SELECT_FROM_MODEL_H
#define SKIGEN_FEATURE_SELECTION_SELECT_FROM_MODEL_H

#include "../Core/Base.h"
#include "../Core/Validation.h"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace Skigen {

namespace detail_sfm {

// Compute per-feature importance: |coef| if 1D, mean(|coef|, axis=0) if 2D.
template <typename Estimator, typename Scalar>
Eigen::Matrix<Scalar, 1, Eigen::Dynamic> compute_importances(
    const Estimator& est) {
    using RowVec = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;
    using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    // Try coef() first. We rely on SFINAE-via-decltype + if-constexpr.
    if constexpr (requires(const Estimator& e) { e.coef(); }) {
        const auto& c = est.coef();
        // c is RowVec or Mat
        using C = std::decay_t<decltype(c)>;
        if constexpr (std::is_same_v<C, RowVec>) {
            return c.array().abs().matrix();
        } else if constexpr (std::is_same_v<C, Mat>) {
            // Mean of absolute values across rows (classes)
            return c.array().abs().colwise().mean().matrix();
        } else {
            // Generic fallback: assume row-vector-like
            RowVec out = c.array().abs().matrix();
            return out;
        }
    } else if constexpr (requires(const Estimator& e) {
                             e.feature_importances();
                         }) {
        return est.feature_importances();
    } else {
        static_assert(sizeof(Estimator) == 0,
                      "Estimator must expose coef() or feature_importances().");
    }
}

// Parse threshold variant. Supports numeric, "mean", "median", "k*mean",
// "k*median", and the special value "-inf"/None equivalent.
template <typename Scalar>
Scalar parse_threshold(
    const std::variant<std::string, Scalar>& threshold,
    const Eigen::Matrix<Scalar, 1, Eigen::Dynamic>& importances) {
    if (std::holds_alternative<Scalar>(threshold)) {
        return std::get<Scalar>(threshold);
    }
    const std::string& s = std::get<std::string>(threshold);

    auto compute_mean = [&]() -> Scalar {
        return importances.mean();
    };
    auto compute_median = [&]() -> Scalar {
        std::vector<Scalar> v(importances.data(),
                              importances.data() + importances.size());
        std::sort(v.begin(), v.end());
        std::size_t n = v.size();
        if (n == 0) return Scalar{0};
        if (n % 2 == 1) return v[n / 2];
        return (v[n / 2 - 1] + v[n / 2]) / Scalar{2};
    };

    // factor*"mean" or factor*"median"
    auto pos = s.find('*');
    if (pos != std::string::npos) {
        std::string factor_str = s.substr(0, pos);
        std::string ref_str = s.substr(pos + 1);
        // Trim whitespace
        auto trim = [](std::string& t) {
            while (!t.empty() && std::isspace(static_cast<unsigned char>(t.front())))
                t.erase(t.begin());
            while (!t.empty() && std::isspace(static_cast<unsigned char>(t.back())))
                t.pop_back();
        };
        trim(factor_str);
        trim(ref_str);
        Scalar factor = static_cast<Scalar>(std::stod(factor_str));
        if (ref_str == "mean") return factor * compute_mean();
        if (ref_str == "median") return factor * compute_median();
        throw std::invalid_argument(
            "SelectFromModel: invalid threshold reference '" + ref_str + "'.");
    }

    if (s == "mean") return compute_mean();
    if (s == "median") return compute_median();

    throw std::invalid_argument(
        "SelectFromModel: invalid threshold string '" + s + "'.");
}

}  // namespace detail_sfm

/// @defgroup Algo_SelectFromModel SelectFromModel
/// @ingroup FeatureSelection
/// @brief Meta-transformer for selecting features based on importance weights.
/// @{

/// @brief Meta-transformer that selects features by an estimator's coefficients
/// or feature importances.
///
/// Mirrors
/// [sklearn.feature_selection.SelectFromModel](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html).
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `estimator` | `Estimator` | (required) | The base estimator with `coef()` or `feature_importances()`. |
/// | `threshold` | `string` or `Scalar` | `"mean"` | Threshold value or string spec like `"median"`, `"1.5*mean"`. |
/// | `prefit` | `bool` | `false` | If `true`, the estimator is assumed already fit and `fit()` skips refitting. |
/// | `max_features` | `optional<int>` | `nullopt` | If set, cap the number of selected features at this value. |
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type | Description |
/// |----------|------|-------------|
/// | `estimator()` | `const Estimator&` | The fitted estimator. |
/// | `threshold_value()` | `Scalar` | The numeric threshold applied to importances. |
/// | `n_features_in()` | `IndexType` | Number of features seen during fit. |
///
/// ### See also
///
/// - Skigen::RFE — Recursive feature elimination.
/// - Skigen::SelectKBest — Top-k features by univariate score.
///
/// ### Limitations relative to scikit-learn `importance_getter` callable is
///   not supported (importance is taken from `coef()`/`feature_importances()`
///   automatically). `norm_order` is not supported (we use absolute value).
template <typename Estimator>
class SelectFromModel {
public:
    using ScalarType = typename Estimator::ScalarType;
    using MatrixType = typename Estimator::MatrixType;
    using VectorType = typename Estimator::VectorType;
    using RowVectorType = typename Estimator::RowVectorType;
    using IndexType = typename Estimator::IndexType;
    using BoolMaskType = Eigen::Array<bool, Eigen::Dynamic, 1>;
    using ThresholdType = std::variant<std::string, ScalarType>;

    explicit SelectFromModel(
        Estimator estimator,
        ThresholdType threshold = std::string{"mean"},
        bool prefit = false,
        std::optional<int> max_features = std::nullopt,
        std::optional<ScalarType> /*importance_getter*/ = std::nullopt)
        : estimator_(std::move(estimator)),
          threshold_(std::move(threshold)),
          prefit_(prefit),
          max_features_(max_features) {}

    [[nodiscard]] const Estimator& estimator() const { return estimator_; }
    [[nodiscard]] Estimator& estimator() { return estimator_; }
    [[nodiscard]] ScalarType threshold_value() const {
        check_is_fitted();
        return threshold_value_;
    }
    [[nodiscard]] bool is_fitted() const noexcept { return fitted_; }
    [[nodiscard]] IndexType n_features_in() const { return n_features_in_; }

    [[nodiscard]] BoolMaskType get_support_mask() const {
        check_is_fitted();
        return support_mask_;
    }
    [[nodiscard]] BoolMaskType get_support(bool /*indices*/ = false) const {
        check_is_fitted();
        return support_mask_;
    }
    [[nodiscard]] Eigen::VectorXi get_support_indices() const {
        check_is_fitted();
        std::vector<int> idx;
        for (Eigen::Index j = 0; j < support_mask_.size(); ++j) {
            if (support_mask_(j)) idx.push_back(static_cast<int>(j));
        }
        Eigen::VectorXi out(idx.size());
        for (std::size_t i = 0; i < idx.size(); ++i) {
            out(static_cast<Eigen::Index>(i)) = idx[i];
        }
        return out;
    }

    template <typename YType>
    SelectFromModel& fit(const Eigen::Ref<const MatrixType>& X,
                         const YType& y) {
        internal::check_non_empty(X);
        n_features_in_ = X.cols();
        if (!prefit_) {
            estimator_.fit(X, y);
        }
        compute_support_from_estimator();
        fitted_ = true;
        return *this;
    }

    [[nodiscard]] MatrixType transform(
        const Eigen::Ref<const MatrixType>& X) const {
        check_is_fitted();
        if (X.cols() != n_features_in_) {
            throw std::invalid_argument(
                "X has " + std::to_string(X.cols()) +
                " features, but SelectFromModel was fitted with " +
                std::to_string(n_features_in_) + " features.");
        }
        Eigen::Index n_kept = 0;
        for (Eigen::Index j = 0; j < support_mask_.size(); ++j) {
            if (support_mask_(j)) ++n_kept;
        }
        MatrixType out(X.rows(), n_kept);
        Eigen::Index k = 0;
        for (Eigen::Index j = 0; j < support_mask_.size(); ++j) {
            if (support_mask_(j)) out.col(k++) = X.col(j);
        }
        return out;
    }

private:
    void check_is_fitted() const {
        if (!fitted_) {
            throw std::runtime_error(
                "SelectFromModel has not been fitted yet. Call fit() first.");
        }
    }

    void compute_support_from_estimator() {
        RowVectorType importances =
            detail_sfm::compute_importances<Estimator, ScalarType>(estimator_);

        if (importances.size() != n_features_in_) {
            throw std::runtime_error(
                "SelectFromModel: importances size mismatch.");
        }

        threshold_value_ =
            detail_sfm::parse_threshold<ScalarType>(threshold_, importances);

        support_mask_ = BoolMaskType::Constant(n_features_in_, false);
        for (Eigen::Index j = 0; j < n_features_in_; ++j) {
            support_mask_(j) = importances(j) >= threshold_value_;
        }

        if (max_features_.has_value()) {
            // Sort selected indices by importance, keep top max_features_.
            int max_k = *max_features_;
            std::vector<Eigen::Index> sel_idx;
            for (Eigen::Index j = 0; j < n_features_in_; ++j) {
                if (support_mask_(j)) sel_idx.push_back(j);
            }
            if (static_cast<int>(sel_idx.size()) > max_k) {
                std::sort(sel_idx.begin(), sel_idx.end(),
                          [&](Eigen::Index a, Eigen::Index b) {
                              return importances(a) > importances(b);
                          });
                support_mask_ = BoolMaskType::Constant(n_features_in_, false);
                for (int i = 0; i < max_k; ++i) {
                    support_mask_(sel_idx[static_cast<std::size_t>(i)]) = true;
                }
            }
        }
    }

    Estimator estimator_;
    ThresholdType threshold_;
    bool prefit_;
    std::optional<int> max_features_;

    bool fitted_ = false;
    IndexType n_features_in_ = 0;
    ScalarType threshold_value_{};
    BoolMaskType support_mask_;
};

/// @}

}  // namespace Skigen

#endif  // SKIGEN_FEATURE_SELECTION_SELECT_FROM_MODEL_H
