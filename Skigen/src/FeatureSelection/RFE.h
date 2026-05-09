// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_FEATURE_SELECTION_RFE_H
#define SKIGEN_FEATURE_SELECTION_RFE_H

#include "../Core/Validation.h"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace Skigen {

namespace detail_rfe {

template <typename Estimator, typename Scalar>
Eigen::Matrix<Scalar, 1, Eigen::Dynamic> compute_importances(
    const Estimator& est) {
    using RowVec = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;
    using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    if constexpr (requires(const Estimator& e) { e.coef(); }) {
        const auto& c = est.coef();
        using C = std::decay_t<decltype(c)>;
        if constexpr (std::is_same_v<C, RowVec>) {
            return c.array().abs().matrix();
        } else if constexpr (std::is_same_v<C, Mat>) {
            return c.array().square().colwise().sum().sqrt().matrix();
        } else {
            RowVec out = c.array().abs().matrix();
            return out;
        }
    } else if constexpr (requires(const Estimator& e) {
                             e.feature_importances();
                         }) {
        return est.feature_importances();
    } else {
        static_assert(sizeof(Estimator) == 0,
                      "RFE: Estimator must expose coef() or "
                      "feature_importances().");
    }
}

}  // namespace detail_rfe

/// @defgroup Algo_RFE RFE
/// @ingroup FeatureSelection
/// @brief Recursive feature elimination wrapper.
/// @{

/// @brief Feature ranking with recursive feature elimination.
///
/// Mirrors
/// [sklearn.feature_selection.RFE](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html).
///
/// Given an external estimator that assigns weights to features, recursive
/// feature elimination (RFE) is to select features by recursively
/// considering smaller and smaller sets of features. First, the estimator
/// is trained on the initial set of features and the importance of each
/// feature is obtained. Then, the least important features are pruned from
/// current set of features. That procedure is recursively repeated on the
/// pruned set until the desired number of features to select is eventually
/// reached.
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `estimator` | `Estimator` | (required) | Base estimator with `coef()` or `feature_importances()`. |
/// | `n_features_to_select` | `int` or `Scalar` | half | Either an absolute count or a fraction in `(0, 1)`. |
/// | `step` | `int` or `Scalar` | `1` | Number (or fraction) of features to remove per iteration. |
/// | `verbose` | `int` | `0` | Verbosity level. |
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type | Description |
/// |----------|------|-------------|
/// | `n_features()` | `int` | Number of selected features. |
/// | `support()` | `Eigen::Array<bool>` | Boolean support mask. |
/// | `ranking()` | `Eigen::VectorXi` | 1-based feature ranks (1 = selected). |
/// | `estimator()` | `const Estimator&` | Estimator refit on selected features. |
///
/// @note **scikit-learn parity gaps:** `importance_getter` callable is not
///   supported. `feature_names_in_` is not exposed.
template <typename Estimator>
class RFE {
public:
    using ScalarType = typename Estimator::ScalarType;
    using MatrixType = typename Estimator::MatrixType;
    using VectorType = typename Estimator::VectorType;
    using RowVectorType = typename Estimator::RowVectorType;
    using IndexType = typename Estimator::IndexType;
    using BoolMaskType = Eigen::Array<bool, Eigen::Dynamic, 1>;
    using NFeaturesType = std::variant<int, ScalarType>;
    using StepType = std::variant<int, ScalarType>;

    explicit RFE(Estimator estimator,
                 std::optional<NFeaturesType> n_features_to_select =
                     std::nullopt,
                 StepType step = 1,
                 int verbose = 0,
                 std::optional<ScalarType> /*importance_getter*/ =
                     std::nullopt)
        : estimator_(std::move(estimator)),
          n_features_to_select_(n_features_to_select),
          step_(std::move(step)),
          verbose_(verbose) {}

    [[nodiscard]] const Estimator& estimator() const { return estimator_; }
    [[nodiscard]] Estimator& estimator() { return estimator_; }
    [[nodiscard]] int n_features() const { check_is_fitted(); return n_selected_; }
    [[nodiscard]] const BoolMaskType& support() const {
        check_is_fitted();
        return support_;
    }
    [[nodiscard]] const Eigen::VectorXi& ranking() const {
        check_is_fitted();
        return ranking_;
    }
    [[nodiscard]] bool is_fitted() const noexcept { return fitted_; }
    [[nodiscard]] IndexType n_features_in() const { return n_features_in_; }

    [[nodiscard]] BoolMaskType get_support(bool /*indices*/ = false) const {
        check_is_fitted();
        return support_;
    }
    [[nodiscard]] Eigen::VectorXi get_support_indices() const {
        check_is_fitted();
        std::vector<int> idx;
        for (Eigen::Index j = 0; j < support_.size(); ++j) {
            if (support_(j)) idx.push_back(static_cast<int>(j));
        }
        Eigen::VectorXi out(idx.size());
        for (std::size_t i = 0; i < idx.size(); ++i) {
            out(static_cast<Eigen::Index>(i)) = idx[i];
        }
        return out;
    }

    template <typename YType>
    RFE& fit(const Eigen::Ref<const MatrixType>& X, const YType& y) {
        internal::check_non_empty(X);
        n_features_in_ = X.cols();
        const int p = static_cast<int>(X.cols());

        // Resolve n_features_to_select.
        int target;
        if (!n_features_to_select_.has_value()) {
            target = p / 2;
            if (target < 1) target = 1;
        } else {
            const NFeaturesType& v = *n_features_to_select_;
            if (std::holds_alternative<int>(v)) {
                target = std::get<int>(v);
            } else {
                ScalarType frac = std::get<ScalarType>(v);
                if (!(frac > ScalarType{0}) || !(frac < ScalarType{1})) {
                    throw std::invalid_argument(
                        "RFE: fractional n_features_to_select must be in (0,1).");
                }
                target = static_cast<int>(std::round(
                    frac * static_cast<ScalarType>(p)));
                if (target < 1) target = 1;
            }
        }
        if (target > p) target = p;
        if (target < 1) target = 1;

        // Resolve step (per-iteration).
        auto compute_step = [&](int n_remaining) -> int {
            if (std::holds_alternative<int>(step_)) {
                int s = std::get<int>(step_);
                if (s <= 0) throw std::invalid_argument(
                    "RFE: integer step must be > 0.");
                return s;
            }
            ScalarType frac = std::get<ScalarType>(step_);
            if (!(frac > ScalarType{0}) || !(frac < ScalarType{1})) {
                throw std::invalid_argument(
                    "RFE: fractional step must be in (0,1).");
            }
            int s = static_cast<int>(
                std::floor(frac * static_cast<ScalarType>(n_remaining)));
            if (s < 1) s = 1;
            return s;
        };

        // Active feature indices.
        std::vector<int> active(static_cast<std::size_t>(p));
        for (int j = 0; j < p; ++j)
            active[static_cast<std::size_t>(j)] = j;

        // Ranking starts at 1 for everyone; eliminated features get higher.
        ranking_ = Eigen::VectorXi::Ones(p);

        while (static_cast<int>(active.size()) > target) {
            // Fit estimator on active features.
            MatrixType X_active(X.rows(), static_cast<Eigen::Index>(active.size()));
            for (std::size_t j = 0; j < active.size(); ++j) {
                X_active.col(static_cast<Eigen::Index>(j)) =
                    X.col(active[j]);
            }
            Estimator clone = estimator_;
            clone.fit(X_active, y);
            RowVectorType imp =
                detail_rfe::compute_importances<Estimator, ScalarType>(clone);

            int to_remove = compute_step(static_cast<int>(active.size()));
            if (static_cast<int>(active.size()) - to_remove < target) {
                to_remove = static_cast<int>(active.size()) - target;
            }
            if (to_remove < 1) to_remove = 1;

            // Determine indices into `active` of features to drop (lowest
            // importance first, sklearn-stable for ties via index).
            std::vector<int> order(active.size());
            for (std::size_t j = 0; j < active.size(); ++j)
                order[j] = static_cast<int>(j);
            std::sort(order.begin(), order.end(),
                      [&](int a, int b) {
                          if (imp(a) == imp(b)) return a < b;
                          return imp(a) < imp(b);
                      });

            // Increment rank of removed features and remove them.
            std::vector<bool> drop(active.size(), false);
            for (int r = 0; r < to_remove; ++r) {
                int idx_in_active = order[static_cast<std::size_t>(r)];
                drop[static_cast<std::size_t>(idx_in_active)] = true;
            }
            // Bump ranks for remaining inactive (already-removed) and the
            // soon-to-be-removed by 1 each iteration.
            for (int j = 0; j < p; ++j) {
                bool in_active = false;
                for (int aj : active) {
                    if (aj == j) { in_active = true; break; }
                }
                if (!in_active) ranking_(j) += 1;
            }
            for (std::size_t j = 0; j < active.size(); ++j) {
                if (drop[j]) ranking_(active[j]) += 1;
            }

            std::vector<int> new_active;
            new_active.reserve(active.size() - static_cast<std::size_t>(to_remove));
            for (std::size_t j = 0; j < active.size(); ++j) {
                if (!drop[j]) new_active.push_back(active[j]);
            }
            active = std::move(new_active);

            if (verbose_ > 0) {
                std::cout << "[RFE] " << active.size() << " features remaining\n";
            }
        }

        // Final fit on selected features.
        n_selected_ = static_cast<int>(active.size());
        support_ = BoolMaskType::Constant(p, false);
        for (int aj : active) support_(aj) = true;

        MatrixType X_final(X.rows(), n_selected_);
        for (int j = 0; j < n_selected_; ++j) {
            X_final.col(j) = X.col(active[static_cast<std::size_t>(j)]);
        }
        estimator_.fit(X_final, y);

        fitted_ = true;
        return *this;
    }

    [[nodiscard]] MatrixType transform(
        const Eigen::Ref<const MatrixType>& X) const {
        check_is_fitted();
        if (X.cols() != n_features_in_) {
            throw std::invalid_argument(
                "X has " + std::to_string(X.cols()) +
                " features, but RFE was fitted with " +
                std::to_string(n_features_in_) + " features.");
        }
        MatrixType out(X.rows(), n_selected_);
        Eigen::Index k = 0;
        for (Eigen::Index j = 0; j < support_.size(); ++j) {
            if (support_(j)) out.col(k++) = X.col(j);
        }
        return out;
    }

    template <typename YType>
    [[nodiscard]] VectorType predict(
        const Eigen::Ref<const MatrixType>& X) const {
        return estimator_.predict(transform(X));
    }

private:
    void check_is_fitted() const {
        if (!fitted_) {
            throw std::runtime_error(
                "RFE has not been fitted yet. Call fit() first.");
        }
    }

    Estimator estimator_;
    std::optional<NFeaturesType> n_features_to_select_;
    StepType step_;
    int verbose_;

    bool fitted_ = false;
    IndexType n_features_in_ = 0;
    int n_selected_ = 0;
    BoolMaskType support_;
    Eigen::VectorXi ranking_;
};

/// @}

}  // namespace Skigen

#endif  // SKIGEN_FEATURE_SELECTION_RFE_H
