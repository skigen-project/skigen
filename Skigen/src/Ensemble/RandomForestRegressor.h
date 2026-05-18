// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_ENSEMBLE_RANDOM_FOREST_REGRESSOR_H
#define SKIGEN_ENSEMBLE_RANDOM_FOREST_REGRESSOR_H

#include "../Core/Base.h"
#include "../Core/Validation.h"
#include "../Tree/DecisionTree.h"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <future>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace Skigen {

/// @addtogroup Algo_RandomForest
/// @{

/// @brief A random forest regressor.
///
/// A meta-estimator that fits a number of decision tree regressors on
/// bootstrap samples of the training set and averages their predictions
/// to improve accuracy and control over-fitting.
///
/// Mirrors
/// [sklearn.ensemble.RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html).
///
/// ### Parameters
///
/// Same as RandomForestClassifier, but with `criterion ∈ {SquaredError,
/// AbsoluteError, FriedmanMSE, Poisson}` (default `SquaredError`) and
/// `max_features_mode = OneThird` (default).
///
/// ### Notes
///
/// The forest prediction is the per-tree mean:
/// @f[ \hat{y}(x) = \frac{1}{T} \sum_{t=1}^T h_t(x). @f]
///
/// ### Limitations relative to scikit-learn Only `criterion=SquaredError` is
///   currently implemented. `min_samples_leaf`,
///   `min_weight_fraction_leaf`, `max_leaf_nodes`,
///   `min_impurity_decrease`, `ccp_alpha`, `monotonic_cst`, `warm_start`
///   are not yet honoured.
template <typename Scalar = double>
class RandomForestRegressor
    : public Predictor<RandomForestRegressor<Scalar>, Scalar> {
public:
    using Base = Predictor<RandomForestRegressor<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;
    using typename Base::ScalarType;

    enum class CriterionReg { SquaredError, AbsoluteError, FriedmanMSE, Poisson };
    enum class MaxFeaturesMode { All, Sqrt, Log2, OneThird, FractionOrCount };

    explicit RandomForestRegressor(
        int n_estimators = 100,
        CriterionReg criterion = CriterionReg::SquaredError,
        std::optional<int> max_depth = std::nullopt,
        int min_samples_split = 2,
        int min_samples_leaf = 1,
        Scalar min_weight_fraction_leaf = Scalar{0},
        MaxFeaturesMode max_features_mode = MaxFeaturesMode::OneThird,
        std::optional<Scalar> max_features_value = std::nullopt,
        std::optional<int> max_leaf_nodes = std::nullopt,
        Scalar min_impurity_decrease = Scalar{0},
        bool bootstrap = true,
        bool oob_score = false,
        int n_jobs = 1,
        std::optional<uint64_t> random_state = std::nullopt,
        int verbose = 0,
        bool warm_start = false,
        std::optional<Scalar> ccp_alpha = std::nullopt,
        std::optional<int> max_samples = std::nullopt)
        : n_estimators_(n_estimators),
          criterion_(criterion),
          max_depth_(max_depth),
          min_samples_split_(min_samples_split),
          min_samples_leaf_(min_samples_leaf),
          min_weight_fraction_leaf_(min_weight_fraction_leaf),
          max_features_mode_(max_features_mode),
          max_features_value_(max_features_value),
          max_leaf_nodes_(max_leaf_nodes),
          min_impurity_decrease_(min_impurity_decrease),
          bootstrap_(bootstrap),
          oob_score_flag_(oob_score),
          n_jobs_(n_jobs),
          random_state_(random_state),
          verbose_(verbose),
          warm_start_(warm_start),
          ccp_alpha_(ccp_alpha),
          max_samples_(max_samples) {
        if (criterion_ != CriterionReg::SquaredError) {
            throw std::invalid_argument(
                "RandomForestRegressor: only criterion=SquaredError is "
                "implemented. AbsoluteError, FriedmanMSE, "
                "Poisson are not implemented.");
        }
    }

    // -- Accessors ---------------------------------------------------------

    [[nodiscard]] int n_estimators() const noexcept { return n_estimators_; }
    [[nodiscard]] bool bootstrap() const noexcept { return bootstrap_; }
    [[nodiscard]] int n_jobs() const noexcept { return n_jobs_; }

    [[nodiscard]] const std::vector<DecisionTreeRegressor<Scalar>>&
    estimators() const { this->check_is_fitted(); return estimators_storage_; }

    [[nodiscard]] const RowVectorType& feature_importances() const {
        this->check_is_fitted(); return feature_importances_;
    }
    [[nodiscard]] const VectorType& oob_prediction() const {
        this->check_is_fitted();
        if (!oob_score_flag_) {
            throw std::runtime_error("oob_prediction_ is only available "
                                     "when constructed with oob_score=true.");
        }
        return oob_prediction_;
    }
    [[nodiscard]] Scalar oob_score() const {
        this->check_is_fitted();
        if (!oob_score_flag_) {
            throw std::runtime_error("oob_score_ is only available when "
                                     "constructed with oob_score=true.");
        }
        return oob_score_value_;
    }

    SKIGEN_PARAMS(
        (n_estimators,             n_estimators_,             int),
        (min_samples_split,        min_samples_split_,        int),
        (min_samples_leaf,         min_samples_leaf_,         int),
        (min_weight_fraction_leaf, min_weight_fraction_leaf_,  double),
        (min_impurity_decrease,    min_impurity_decrease_,     double),
        (bootstrap,                bootstrap_,                bool),
        (oob_score,                oob_score_flag_,           bool),
        (n_jobs,                   n_jobs_,                   int),
        (verbose,                  verbose_,                  int),
        (warm_start,               warm_start_,               bool))

    // -- Fit/Predict --------------------------------------------------------

    RandomForestRegressor& fit_impl(const Eigen::Ref<const MatrixType>& X,
                                    const Eigen::Ref<const VectorType>& y) {
        internal::check_non_empty(X);
        internal::check_consistent_length(X, y);

        this->n_features_in_ = X.cols();
        const Eigen::Index n_samples = X.rows();
        const int sample_count = max_samples_.has_value()
            ? std::min<int>(*max_samples_, static_cast<int>(n_samples))
            : static_cast<int>(n_samples);

        const uint64_t base_seed = random_state_.value_or(
            static_cast<uint64_t>(std::random_device{}()));

        std::vector<std::vector<Eigen::Index>> per_tree_indices(
            static_cast<std::size_t>(n_estimators_));
        for (int t = 0; t < n_estimators_; ++t) {
            std::mt19937_64 rng(base_seed ^ static_cast<uint64_t>(t));
            std::vector<Eigen::Index> idx;
            idx.reserve(static_cast<std::size_t>(sample_count));
            if (bootstrap_) {
                std::uniform_int_distribution<Eigen::Index> dist(0, n_samples - 1);
                for (int i = 0; i < sample_count; ++i) idx.push_back(dist(rng));
            } else {
                idx.resize(static_cast<std::size_t>(n_samples));
                std::iota(idx.begin(), idx.end(), Eigen::Index{0});
            }
            per_tree_indices[static_cast<std::size_t>(t)] = std::move(idx);
        }

        // Translate ensemble MaxFeaturesMode to the tree's integer mode.
        // Tree mode codes: 0=All, 1=Sqrt, 2=Log2, 3=FractionOrCount.
        int tree_mf_mode = 0;
        double tree_mf_value = 0.0;
        switch (max_features_mode_) {
            case MaxFeaturesMode::All:    tree_mf_mode = 0; break;
            case MaxFeaturesMode::Sqrt:   tree_mf_mode = 1; break;
            case MaxFeaturesMode::Log2:   tree_mf_mode = 2; break;
            case MaxFeaturesMode::OneThird:
                tree_mf_mode = 3;
                tree_mf_value = 1.0 / 3.0;
                break;
            case MaxFeaturesMode::FractionOrCount:
                tree_mf_mode = 3;
                tree_mf_value = max_features_value_.has_value()
                    ? static_cast<double>(*max_features_value_) : 1.0;
                break;
        }

        const int tree_max_depth = max_depth_.value_or(-1);

        estimators_storage_.clear();
        estimators_storage_.reserve(static_cast<std::size_t>(n_estimators_));
        for (int t = 0; t < n_estimators_; ++t) {
            uint64_t tree_seed = base_seed ^
                (static_cast<uint64_t>(t) * 0x9E3779B97F4A7C15ULL);
            estimators_storage_.emplace_back(
                tree_max_depth, min_samples_split_, tree_mf_mode, tree_mf_value,
                std::optional<uint64_t>(tree_seed));
        }

        auto fit_one = [&](int t) {
            estimators_storage_[static_cast<std::size_t>(t)]
                .fit_with_indices(X, y, per_tree_indices[static_cast<std::size_t>(t)]);
        };

        if (n_jobs_ > 1) {
            const int budget = std::min(n_jobs_, n_estimators_);
            int next = 0;
            while (next < n_estimators_) {
                int batch = std::min(budget, n_estimators_ - next);
                std::vector<std::future<void>> futs;
                futs.reserve(static_cast<std::size_t>(batch));
                for (int b = 0; b < batch; ++b) {
                    int idx = next + b;
                    futs.emplace_back(std::async(std::launch::async,
                                                 [fit_one, idx]() { fit_one(idx); }));
                }
                for (auto& f : futs) f.get();
                next += batch;
            }
        } else {
            for (int t = 0; t < n_estimators_; ++t) fit_one(t);
        }

        // Mean of per-tree feature importances.
        feature_importances_ = RowVectorType::Zero(X.cols());
        for (const auto& tree : estimators_storage_)
            feature_importances_ += tree.feature_importances();
        feature_importances_ /= static_cast<Scalar>(n_estimators_);
        Scalar fi_sum = feature_importances_.sum();
        if (fi_sum > Scalar{0}) feature_importances_ /= fi_sum;

        if (oob_score_flag_) {
            compute_oob(X, y, per_tree_indices);
        }

        this->fitted_ = true;
        return *this;
    }

    [[nodiscard]] VectorType predict_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        VectorType acc = VectorType::Zero(X.rows());
        for (const auto& tree : estimators_storage_) {
            acc += tree.predict(X);
        }
        acc /= static_cast<Scalar>(n_estimators_);
        return acc;
    }

    [[nodiscard]] ScalarType score_impl(
        const Eigen::Ref<const MatrixType>& X,
        const Eigen::Ref<const VectorType>& y) const {
        VectorType y_pred = predict_impl(X);
        Scalar ss_res = (y - y_pred).squaredNorm();
        Scalar ss_tot = (y.array() - y.mean()).matrix().squaredNorm();
        if (ss_tot == Scalar{0}) return Scalar{0};
        return Scalar{1} - ss_res / ss_tot;
    }

private:
    int n_estimators_;
    CriterionReg criterion_;
    std::optional<int> max_depth_;
    int min_samples_split_;
    int min_samples_leaf_;
    Scalar min_weight_fraction_leaf_;
    MaxFeaturesMode max_features_mode_;
    std::optional<Scalar> max_features_value_;
    std::optional<int> max_leaf_nodes_;
    Scalar min_impurity_decrease_;
    bool bootstrap_;
    bool oob_score_flag_;
    int n_jobs_;
    std::optional<uint64_t> random_state_;
    int verbose_;
    bool warm_start_;
    std::optional<Scalar> ccp_alpha_;
    std::optional<int> max_samples_;

    std::vector<DecisionTreeRegressor<Scalar>> estimators_storage_;
    RowVectorType feature_importances_;
    VectorType oob_prediction_;
    Scalar oob_score_value_{0};

    void compute_oob(const Eigen::Ref<const MatrixType>& X,
                     const Eigen::Ref<const VectorType>& y,
                     const std::vector<std::vector<Eigen::Index>>& per_tree_indices) {
        const Eigen::Index n_samples = X.rows();
        oob_prediction_ = VectorType::Zero(n_samples);
        Eigen::VectorXi counts = Eigen::VectorXi::Zero(n_samples);

        for (std::size_t t = 0; t < estimators_storage_.size(); ++t) {
            std::vector<bool> in_bag(static_cast<std::size_t>(n_samples), false);
            for (auto idx : per_tree_indices[t])
                in_bag[static_cast<std::size_t>(idx)] = true;

            std::vector<Eigen::Index> oob_rows;
            for (Eigen::Index i = 0; i < n_samples; ++i)
                if (!in_bag[static_cast<std::size_t>(i)]) oob_rows.push_back(i);
            if (oob_rows.empty()) continue;

            MatrixType X_oob(oob_rows.size(), X.cols());
            for (std::size_t k = 0; k < oob_rows.size(); ++k)
                X_oob.row(static_cast<Eigen::Index>(k)) = X.row(oob_rows[k]);

            VectorType pred = estimators_storage_[t].predict(X_oob);
            for (std::size_t k = 0; k < oob_rows.size(); ++k) {
                Eigen::Index r = oob_rows[k];
                oob_prediction_(r) += pred(static_cast<Eigen::Index>(k));
                counts(r) += 1;
            }
        }

        Scalar ss_res{0};
        Scalar y_mean{0};
        int counted = 0;
        for (Eigen::Index i = 0; i < n_samples; ++i) {
            if (counts(i) > 0) {
                oob_prediction_(i) /= static_cast<Scalar>(counts(i));
                y_mean += y(i);
                ++counted;
            } else {
                oob_prediction_(i) = std::numeric_limits<Scalar>::quiet_NaN();
            }
        }
        if (counted == 0) { oob_score_value_ = Scalar{0}; return; }
        y_mean /= static_cast<Scalar>(counted);

        Scalar ss_tot{0};
        for (Eigen::Index i = 0; i < n_samples; ++i) {
            if (counts(i) > 0) {
                Scalar d = y(i) - oob_prediction_(i);
                ss_res += d * d;
                Scalar dy = y(i) - y_mean;
                ss_tot += dy * dy;
            }
        }
        oob_score_value_ = ss_tot > Scalar{0}
            ? Scalar{1} - ss_res / ss_tot : Scalar{0};
    }
};

/// @}

} // namespace Skigen

#endif // SKIGEN_ENSEMBLE_RANDOM_FOREST_REGRESSOR_H
