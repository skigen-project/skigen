// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_MODEL_SELECTION_RANDOMIZED_SEARCH_CV_H
#define SKIGEN_MODEL_SELECTION_RANDOMIZED_SEARCH_CV_H

#include "../Core/Params.h"
#include "CrossValidation.h"
#include "ParameterGrid.h"

#include <Eigen/Core>

#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace Skigen {

/// @addtogroup ModelSelection
/// @{

/// @brief Randomised search over a parameter grid with cross-validation.
///
/// Mirrors
/// [sklearn.model_selection.RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html).
///
/// Samples `n_iter` random parameter combinations from the supplied grid
/// (with replacement) and evaluates each via K-fold cross-validation.
/// Unlike `GridSearchCV`, only a fixed budget of combinations is tried.
///
/// ### Parameters
///
/// | Parameter | Type | Default |
/// |---|---|---|
/// | `estimator` | `Estimator` | (required) |
/// | `param_distributions` | `ParameterGrid` | (required; discrete lists) |
/// | `n_iter` | `int` | `10` |
/// | `cv` | `int` | `5` |
/// | `refit` | `bool` | `true` |
/// | `random_state` | `optional<uint64_t>` | `nullopt` |
template <typename Estimator, typename Scalar = double>
class RandomizedSearchCV {
public:
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    explicit RandomizedSearchCV(
        Estimator estimator,
        ParameterGrid param_distributions,
        int n_iter = 10,
        int cv = 5,
        bool refit = true,
        std::optional<uint64_t> random_state = std::nullopt)
        : estimator_(std::move(estimator)),
          param_dist_(std::move(param_distributions)),
          n_iter_(n_iter),
          cv_(cv),
          refit_(refit),
          random_state_(random_state) {}

    template <typename YType>
    RandomizedSearchCV& fit(const MatrixType& X, const YType& y) {
        const auto& grid = param_dist_.grid();
        if (grid.empty()) {
            throw std::invalid_argument(
                "RandomizedSearchCV.fit: empty parameter distributions.");
        }

        const uint64_t seed = random_state_.value_or(0ULL);
        std::mt19937_64 rng(seed);

        cv_params_.clear();
        cv_params_.reserve(static_cast<std::size_t>(n_iter_));
        cv_mean_scores_.clear();
        cv_mean_scores_.reserve(static_cast<std::size_t>(n_iter_));

        best_score_ = -std::numeric_limits<Scalar>::infinity();
        best_index_ = 0;

        for (int it = 0; it < n_iter_; ++it) {
            ParameterDict params;
            for (const auto& [name, vals] : grid) {
                std::uniform_int_distribution<std::size_t> dist(
                    0, vals.size() - 1);
                params[name] = vals[dist(rng)];
            }
            cv_params_.push_back(params);

            Estimator est = estimator_;
            for (const auto& [name, val] : params) {
                est.set_param(name, val);
            }

            auto scores = cross_val_score<Estimator, Scalar>(
                est, X, y, cv_);
            const Scalar mean = scores.mean();
            cv_mean_scores_.push_back(mean);

            if (mean > best_score_) {
                best_score_ = mean;
                best_index_ = static_cast<std::size_t>(it);
            }
        }

        if (refit_) {
            best_estimator_ = estimator_;
            for (const auto& [name, val] : cv_params_[best_index_]) {
                best_estimator_.set_param(name, val);
            }
            best_estimator_.fit(X, y);
        }

        fitted_ = true;
        return *this;
    }

    template <typename XType>
    [[nodiscard]] auto predict(const XType& X) const {
        check_fitted();
        return best_estimator_.predict(X);
    }

    template <typename XType, typename YType>
    [[nodiscard]] Scalar score(const XType& X, const YType& y) const {
        check_fitted();
        return best_estimator_.score(X, y);
    }

    [[nodiscard]] const Estimator& best_estimator() const {
        check_fitted(); return best_estimator_;
    }
    [[nodiscard]] ParameterDict best_params() const {
        check_fitted(); return cv_params_[best_index_];
    }
    [[nodiscard]] Scalar best_score() const {
        check_fitted(); return best_score_;
    }
    [[nodiscard]] std::size_t best_index() const {
        check_fitted(); return best_index_;
    }
    [[nodiscard]] const std::vector<ParameterDict>& cv_results_params() const {
        check_fitted(); return cv_params_;
    }
    [[nodiscard]] const std::vector<Scalar>& cv_results_mean_score() const {
        check_fitted(); return cv_mean_scores_;
    }

private:
    void check_fitted() const {
        if (!fitted_) {
            throw std::runtime_error(
                "RandomizedSearchCV: not fitted. Call fit() first.");
        }
    }

    Estimator estimator_;
    ParameterGrid param_dist_;
    int n_iter_;
    int cv_;
    bool refit_;
    std::optional<uint64_t> random_state_;

    bool fitted_{false};
    Estimator best_estimator_{estimator_};
    Scalar best_score_{0};
    std::size_t best_index_{0};
    std::vector<ParameterDict> cv_params_;
    std::vector<Scalar> cv_mean_scores_;
};

/// @}

}  // namespace Skigen

#endif  // SKIGEN_MODEL_SELECTION_RANDOMIZED_SEARCH_CV_H
