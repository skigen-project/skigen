// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

#ifndef SKIGEN_IMPUTE_SIMPLE_IMPUTER_H
#define SKIGEN_IMPUTE_SIMPLE_IMPUTER_H

#include "../Core/Base.h"
#include "../Core/Validation.h"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace Skigen {

/// @defgroup Algo_Impute Imputation
/// @ingroup Impute
/// @brief Missing-value imputation and missingness indicators.
/// @{

/// @brief Strategy used by SimpleImputer.
enum class ImputeStrategy {
    Mean,
    Median,
    MostFrequent,
    Constant
};

/// @brief Feature-selection mode used by MissingIndicator.
enum class MissingIndicatorFeatures {
    MissingOnly,
    All
};

namespace internal {

template <typename Scalar>
[[nodiscard]] bool is_missing_value(Scalar value, Scalar missing_value) {
    if (std::isnan(missing_value)) {
        return std::isnan(value);
    }
    return value == missing_value;
}

template <typename Scalar>
[[nodiscard]] Scalar median_sorted(std::vector<Scalar>& values) {
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    if (n % 2 == 1) {
        return values[n / 2];
    }
    return (values[n / 2 - 1] + values[n / 2]) / Scalar{2};
}

template <typename Scalar>
[[nodiscard]] Scalar most_frequent_sorted(std::vector<Scalar>& values) {
    std::sort(values.begin(), values.end());
    Scalar best_value = values.front();
    std::size_t best_count = 1;
    Scalar current_value = values.front();
    std::size_t current_count = 1;

    for (std::size_t i = 1; i < values.size(); ++i) {
        if (values[i] == current_value) {
            ++current_count;
            continue;
        }
        if (current_count > best_count) {
            best_count = current_count;
            best_value = current_value;
        }
        current_value = values[i];
        current_count = 1;
    }
    if (current_count > best_count) {
        best_value = current_value;
    }
    return best_value;
}

}  // namespace internal

/// @brief Binary mask transformer for missing values.
///
/// Mirrors the dense numeric subset of
/// [sklearn.impute.MissingIndicator](https://scikit-learn.org/stable/modules/generated/sklearn.impute.MissingIndicator.html).
///
/// The returned mask is a numeric matrix containing `0` and `1`, matching
/// Skigen's dense `Transformer` return type.
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `missing_values` | `Scalar` | `NaN` | Marker treated as missing. |
/// | `features` | `MissingIndicatorFeatures` | `MissingOnly` | Emit only features missing at fit time, or all features. |
/// | `error_on_new` | `bool` | `true` | Raise when transform data has missing values in previously complete features. |
///
/// ### Examples
///
/// @snippet simple_imputer.cpp example_missing_indicator
template <typename Scalar = double>
class MissingIndicator : public Transformer<MissingIndicator<Scalar>, Scalar> {
public:
    using Base = Transformer<MissingIndicator<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;

    explicit MissingIndicator(
        Scalar missing_values = std::numeric_limits<Scalar>::quiet_NaN(),
        MissingIndicatorFeatures features = MissingIndicatorFeatures::MissingOnly,
        bool error_on_new = true)
        : missing_values_(missing_values),
          features_(features),
          error_on_new_(error_on_new) {}

    [[nodiscard]] Scalar missing_values() const noexcept { return missing_values_; }
    [[nodiscard]] MissingIndicatorFeatures features() const noexcept { return features_; }
    [[nodiscard]] bool error_on_new() const noexcept { return error_on_new_; }

    [[nodiscard]] const Eigen::VectorXi& features_indices() const {
        this->check_is_fitted();
        return features_indices_;
    }

    MissingIndicator& fit_impl(const Eigen::Ref<const MatrixType>& X) {
        internal::check_non_empty(X);
        this->n_features_in_ = X.cols();
        std::vector<int> selected;
        selected.reserve(static_cast<std::size_t>(X.cols()));

        for (IndexType col = 0; col < X.cols(); ++col) {
            bool has_missing = false;
            for (IndexType row = 0; row < X.rows(); ++row) {
                if (internal::is_missing_value(X(row, col), missing_values_)) {
                    has_missing = true;
                    break;
                }
            }
            if (features_ == MissingIndicatorFeatures::All || has_missing) {
                selected.push_back(static_cast<int>(col));
            }
        }
        features_indices_.resize(static_cast<IndexType>(selected.size()));
        for (IndexType i = 0; i < features_indices_.size(); ++i) {
            features_indices_(i) = selected[static_cast<std::size_t>(i)];
        }
        this->fitted_ = true;
        return *this;
    }

    [[nodiscard]] MatrixType transform_impl(const Eigen::Ref<const MatrixType>& X) const {
        if (features_ == MissingIndicatorFeatures::MissingOnly && error_on_new_) {
            for (IndexType col = 0; col < X.cols(); ++col) {
                bool tracked = false;
                for (IndexType i = 0; i < features_indices_.size(); ++i) {
                    if (features_indices_(i) == col) {
                        tracked = true;
                        break;
                    }
                }
                if (tracked) continue;
                for (IndexType row = 0; row < X.rows(); ++row) {
                    if (internal::is_missing_value(X(row, col), missing_values_)) {
                        throw std::invalid_argument(
                            "MissingIndicator: transform data has missing values "
                            "in a feature that had no missing values during fit.");
                    }
                }
            }
        }

        MatrixType mask = MatrixType::Zero(X.rows(), features_indices_.size());
        for (IndexType out_col = 0; out_col < features_indices_.size(); ++out_col) {
            const IndexType source_col = features_indices_(out_col);
            for (IndexType row = 0; row < X.rows(); ++row) {
                mask(row, out_col) = internal::is_missing_value(X(row, source_col), missing_values_)
                    ? Scalar{1}
                    : Scalar{0};
            }
        }
        return mask;
    }

private:
    Scalar missing_values_;
    MissingIndicatorFeatures features_;
    bool error_on_new_;
    Eigen::VectorXi features_indices_;
};

/// @brief Univariate missing-value imputer.
///
/// Mirrors the dense numeric subset of
/// [sklearn.impute.SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html).
///
/// Dense matrices are imputed feature-wise. When `add_indicator=true`, the
/// transformed output appends indicator columns for features that contained
/// missing values during fitting.
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `missing_values` | `Scalar` | `NaN` | Marker treated as missing. |
/// | `strategy` | `ImputeStrategy` | `Mean` | Statistic used for replacement. |
/// | `fill_value` | `Scalar` | `0` | Constant used by `Constant`. |
/// | `add_indicator` | `bool` | `false` | Append missingness indicators for fit-time missing features. |
/// | `keep_empty_features` | `bool` | `false` | Retain all-missing features and fill them with zero or `fill_value`. |
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type | Description |
/// |----------|------|-------------|
/// | `statistics()` | `RowVectorType` | Per-feature imputation statistics. |
/// | `indicator_features()` | `Eigen::VectorXi` | Features represented by appended indicator columns. |
///
/// ### Limitations relative to scikit-learn
///
/// This first Skigen implementation supports dense numeric input. Sparse input,
/// string/categorical values, callable strategies, feature names, `copy`, and
/// `inverse_transform()` are deferred.
///
/// ### Examples
///
/// @snippet simple_imputer.cpp example_simple_imputer
template <typename Scalar = double>
class SimpleImputer : public Transformer<SimpleImputer<Scalar>, Scalar> {
public:
    using Base = Transformer<SimpleImputer<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;

    explicit SimpleImputer(
        Scalar missing_values = std::numeric_limits<Scalar>::quiet_NaN(),
        ImputeStrategy strategy = ImputeStrategy::Mean,
        Scalar fill_value = Scalar{0},
        bool add_indicator = false,
        bool keep_empty_features = false)
        : missing_values_(missing_values),
          strategy_(strategy),
          fill_value_(fill_value),
          add_indicator_(add_indicator),
          keep_empty_features_(keep_empty_features) {}

    [[nodiscard]] Scalar missing_values() const noexcept { return missing_values_; }
    [[nodiscard]] ImputeStrategy strategy() const noexcept { return strategy_; }
    [[nodiscard]] Scalar fill_value() const noexcept { return fill_value_; }
    [[nodiscard]] bool add_indicator() const noexcept { return add_indicator_; }
    [[nodiscard]] bool keep_empty_features() const noexcept { return keep_empty_features_; }

    [[nodiscard]] const RowVectorType& statistics() const {
        this->check_is_fitted();
        return statistics_;
    }

    [[nodiscard]] const Eigen::VectorXi& indicator_features() const {
        this->check_is_fitted();
        return indicator_features_;
    }

    SimpleImputer& fit_impl(const Eigen::Ref<const MatrixType>& X) {
        internal::check_non_empty(X);
        this->n_features_in_ = X.cols();
        statistics_.resize(X.cols());
        drop_features_.assign(static_cast<std::size_t>(X.cols()), false);
        std::vector<int> indicator_features;
        indicator_features.reserve(static_cast<std::size_t>(X.cols()));

        for (IndexType col = 0; col < X.cols(); ++col) {
            std::vector<Scalar> observed;
            observed.reserve(static_cast<std::size_t>(X.rows()));
            bool had_missing = false;
            for (IndexType row = 0; row < X.rows(); ++row) {
                if (internal::is_missing_value(X(row, col), missing_values_)) {
                    had_missing = true;
                } else {
                    observed.push_back(X(row, col));
                }
            }

            if (had_missing) {
                indicator_features.push_back(static_cast<int>(col));
            }

            if (observed.empty()) {
                if (keep_empty_features_) {
                    statistics_(col) = strategy_ == ImputeStrategy::Constant ? fill_value_ : Scalar{0};
                } else {
                    statistics_(col) = std::numeric_limits<Scalar>::quiet_NaN();
                    drop_features_[static_cast<std::size_t>(col)] = true;
                }
                continue;
            }

            switch (strategy_) {
                case ImputeStrategy::Mean: {
                    Scalar sum = Scalar{0};
                    for (const Scalar value : observed) sum += value;
                    statistics_(col) = sum / static_cast<Scalar>(observed.size());
                    break;
                }
                case ImputeStrategy::Median:
                    statistics_(col) = internal::median_sorted(observed);
                    break;
                case ImputeStrategy::MostFrequent:
                    statistics_(col) = internal::most_frequent_sorted(observed);
                    break;
                case ImputeStrategy::Constant:
                    statistics_(col) = fill_value_;
                    break;
            }
        }

        indicator_features_.resize(static_cast<IndexType>(indicator_features.size()));
        for (IndexType i = 0; i < indicator_features_.size(); ++i) {
            indicator_features_(i) = indicator_features[static_cast<std::size_t>(i)];
        }

        this->fitted_ = true;
        return *this;
    }

    [[nodiscard]] MatrixType transform_impl(const Eigen::Ref<const MatrixType>& X) const {
        IndexType kept_features = 0;
        for (IndexType col = 0; col < X.cols(); ++col) {
            if (!drop_features_[static_cast<std::size_t>(col)]) ++kept_features;
        }

        const IndexType indicator_count = add_indicator_ ? indicator_features_.size() : 0;
        MatrixType out(X.rows(), kept_features + indicator_count);

        IndexType out_col = 0;
        for (IndexType col = 0; col < X.cols(); ++col) {
            if (drop_features_[static_cast<std::size_t>(col)]) continue;
            const Scalar statistic = statistics_(col);
            for (IndexType row = 0; row < X.rows(); ++row) {
                out(row, out_col) = internal::is_missing_value(X(row, col), missing_values_)
                    ? statistic
                    : X(row, col);
            }
            ++out_col;
        }

        if (add_indicator_) {
            for (IndexType indicator = 0; indicator < indicator_features_.size(); ++indicator) {
                const IndexType source_col = indicator_features_(indicator);
                for (IndexType row = 0; row < X.rows(); ++row) {
                    out(row, out_col + indicator) =
                        internal::is_missing_value(X(row, source_col), missing_values_) ? Scalar{1} : Scalar{0};
                }
            }
        }

        return out;
    }

private:
    Scalar missing_values_;
    ImputeStrategy strategy_;
    Scalar fill_value_;
    bool add_indicator_;
    bool keep_empty_features_;
    RowVectorType statistics_;
    Eigen::VectorXi indicator_features_;
    std::vector<bool> drop_features_;
};

/// @}

}  // namespace Skigen

#endif  // SKIGEN_IMPUTE_SIMPLE_IMPUTER_H
