// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_MODEL_SELECTION_GRID_SEARCH_CV_H
#define SKIGEN_MODEL_SELECTION_GRID_SEARCH_CV_H

#include "../Core/Params.h"
#include "CrossValidation.h"
#include "ParameterGrid.h"

#include <Eigen/Core>

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace Skigen {

/// @addtogroup ModelSelection
/// @{

/// @brief Exhaustive search over a parameter grid with cross-validation.
///
/// Mirrors
/// [sklearn.model_selection.GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).
///
/// Fits the estimator for every point in the Cartesian product of the
/// supplied parameter grid, evaluates each via K-fold cross-validation,
/// and optionally refits the best configuration on the full training set.
///
/// ### Parameters
///
/// | Parameter | Type | Default |
/// |---|---|---|
/// | `estimator` | `Estimator` | (required) |
/// | `param_grid` | `ParameterGrid` | (required) |
/// | `cv` | `int` | `5` |
/// | `refit` | `bool` | `true` |
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type |
/// |---|---|
/// | `best_estimator()` | `const Estimator&` |
/// | `best_params()` | `ParameterDict` |
/// | `best_score()` | `Scalar` |
/// | `best_index()` | `std::size_t` |
/// | `cv_results_params()` | `vector<ParameterDict>` |
/// | `cv_results_mean_score()` | `vector<Scalar>` |
template <typename Estimator, typename Scalar = double>
class GridSearchCV {
public:
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    explicit GridSearchCV(
        Estimator estimator,
        ParameterGrid param_grid,
        int cv = 5,
        bool refit = true)
        : estimator_(std::move(estimator)),
          param_grid_(std::move(param_grid)),
          cv_(cv),
          refit_(refit) {}

    template <typename YType>
    GridSearchCV& fit(const MatrixType& X, const YType& y) {
        const std::size_t n_combos = param_grid_.size();
        if (n_combos == 0) {
            throw std::invalid_argument(
                "GridSearchCV.fit: empty parameter grid.");
        }

        cv_params_.clear();
        cv_params_.reserve(n_combos);
        cv_mean_scores_.clear();
        cv_mean_scores_.reserve(n_combos);

        best_score_ = -std::numeric_limits<Scalar>::infinity();
        best_index_ = 0;

        for (std::size_t i = 0; i < n_combos; ++i) {
            ParameterDict params = param_grid_[i];
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
                best_index_ = i;
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
                "GridSearchCV: not fitted. Call fit() first.");
        }
    }

    Estimator estimator_;
    ParameterGrid param_grid_;
    int cv_;
    bool refit_;

    bool fitted_{false};
    Estimator best_estimator_{estimator_};
    Scalar best_score_{0};
    std::size_t best_index_{0};
    std::vector<ParameterDict> cv_params_;
    std::vector<Scalar> cv_mean_scores_;
};

/// @}

}  // namespace Skigen

#endif  // SKIGEN_MODEL_SELECTION_GRID_SEARCH_CV_H
