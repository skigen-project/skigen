// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_ENSEMBLE_RANDOM_FOREST_CLASSIFIER_H
#define SKIGEN_ENSEMBLE_RANDOM_FOREST_CLASSIFIER_H

#include "../Core/Base.h"
#include "../Core/Validation.h"
#include "../Tree/DecisionTree.h"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <future>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace Skigen {

/// @defgroup Algo_RandomForest Random Forest
/// @ingroup Ensemble
/// @brief Bagged ensembles of decision trees.
/// @{

/// @brief A random forest classifier.
///
/// A meta-estimator that fits a number of decision tree classifiers on
/// various sub-samples of the dataset and uses averaging to improve the
/// predictive accuracy and control over-fitting. Each tree is trained on
/// a bootstrap sample of the training data and considers only a random
/// subset of features at each split.
///
/// Mirrors
/// [sklearn.ensemble.RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `n_estimators` | `int` | `100` | Number of trees in the forest. |
/// | `criterion` | `CriterionClf` | `Gini` | Split quality function. |
/// | `max_depth` | `optional<int>` | `nullopt` | Max tree depth (unlimited if absent). |
/// | `min_samples_split` | `int` | `2` | Min samples required to split a node. |
/// | `min_samples_leaf` | `int` | `1` | Min samples per leaf (sklearn parity gap). |
/// | `min_weight_fraction_leaf` | `Scalar` | `0` | **Deprecated**, no-op. |
/// | `max_features_mode` | `MaxFeaturesMode` | `Sqrt` | Feature subspace mode at each split. |
/// | `max_features_value` | `optional<Scalar>` | `nullopt` | Used with `FractionOrCount`. |
/// | `max_leaf_nodes` | `optional<int>` | `nullopt` | sklearn parity gap. |
/// | `min_impurity_decrease` | `Scalar` | `0` | sklearn parity gap. |
/// | `bootstrap` | `bool` | `true` | Whether bootstrap samples are used. |
/// | `oob_score` | `bool` | `false` | Use out-of-bag samples to estimate generalization score. |
/// | `n_jobs` | `int` | `1` | Number of parallel jobs (uses `std::async`). |
/// | `random_state` | `optional<uint64_t>` | `nullopt` | RNG seed. |
/// | `verbose` | `int` | `0` | Verbosity (currently unused). |
/// | `warm_start` | `bool` | `false` | sklearn parity gap. |
/// | `ccp_alpha` | `optional<Scalar>` | `nullopt` | sklearn parity gap. |
/// | `max_samples` | `optional<int>` | `nullopt` | Bootstrap sample size (defaults to n_samples). |
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type | Description |
/// |----------|------|-------------|
/// | `estimators()` | `vector<DecisionTreeClassifier>` | Fitted base trees. |
/// | `classes()` | `Eigen::VectorXi` | Class labels (sorted). |
/// | `n_classes()` | `int` | Number of classes. |
/// | `n_features_in_()` | `int` | Number of features seen during fit. |
/// | `feature_importances()` | `RowVectorType` | Mean Gini-based importances. |
/// | `oob_decision_function()` | `MatrixType` | OOB class probability matrix. |
/// | `oob_score()` | `Scalar` | OOB accuracy. |
///
/// ### Notes
///
/// Each tree fits a bootstrap sample of size `max_samples` (defaults to
/// `n_samples`). At each node, a random subset of features of size
/// @f$ m_{eff} @f$ is considered for splitting. The forest prediction is
/// the argmax of the class probability average over trees:
/// @f[
///   \hat{y} = \arg\max_c \frac{1}{T} \sum_{t=1}^T P_t(c \mid x).
/// @f]
///
/// @note **scikit-learn parity gaps:** `criterion::LogLoss` is not yet
///   implemented. `min_samples_leaf`, `min_weight_fraction_leaf`,
///   `max_leaf_nodes`, `min_impurity_decrease`, `class_weight`,
///   `ccp_alpha`, `monotonic_cst`, `warm_start` are not yet honoured.
template <typename Scalar = double>
class RandomForestClassifier
    : public Classifier<RandomForestClassifier<Scalar>, Scalar> {
public:
    using Base = Classifier<RandomForestClassifier<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;
    using typename Base::ScalarType;
    using typename Base::LabelType;

    enum class CriterionClf { Gini, Entropy, LogLoss };
    enum class MaxFeaturesMode { All, Sqrt, Log2, FractionOrCount };

    explicit RandomForestClassifier(
        int n_estimators = 100,
        CriterionClf criterion = CriterionClf::Gini,
        std::optional<int> max_depth = std::nullopt,
        int min_samples_split = 2,
        int min_samples_leaf = 1,
        Scalar min_weight_fraction_leaf = Scalar{0},
        MaxFeaturesMode max_features_mode = MaxFeaturesMode::Sqrt,
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
        // Reject unsupported configurations early (matches sklearn parity-gap pattern).
        if (criterion_ == CriterionClf::LogLoss) {
            throw std::invalid_argument(
                "RandomForestClassifier: criterion=LogLoss not yet implemented in "
                "Skigen v1.1.0. Use Gini or Entropy.");
        }
    }

    // -- Accessors ---------------------------------------------------------

    [[nodiscard]] int n_estimators() const noexcept { return n_estimators_; }
    [[nodiscard]] bool bootstrap() const noexcept { return bootstrap_; }
    [[nodiscard]] int n_jobs() const noexcept { return n_jobs_; }

    [[nodiscard]] const std::vector<DecisionTreeClassifier<Scalar>>&
    estimators() const { this->check_is_fitted(); return estimators_storage_; }

    [[nodiscard]] const Eigen::VectorXi& classes() const {
        this->check_is_fitted(); return classes_;
    }
    [[nodiscard]] int n_classes() const {
        this->check_is_fitted(); return n_classes_val_;
    }
    [[nodiscard]] const RowVectorType& feature_importances() const {
        this->check_is_fitted(); return feature_importances_;
    }
    [[nodiscard]] const MatrixType& oob_decision_function() const {
        this->check_is_fitted();
        if (!oob_score_flag_) {
            throw std::runtime_error("oob_decision_function_ is only available "
                                     "when constructed with oob_score=true.");
        }
        return oob_decision_function_;
    }
    [[nodiscard]] Scalar oob_score() const {
        this->check_is_fitted();
        if (!oob_score_flag_) {
            throw std::runtime_error("oob_score_ is only available when "
                                     "constructed with oob_score=true.");
        }
        return oob_score_value_;
    }

    // -- Fit/Predict implementation ----------------------------------------

    RandomForestClassifier& fit_impl(
        const Eigen::Ref<const MatrixType>& X,
        const Eigen::Ref<const Eigen::VectorXi>& y) {
        internal::check_non_empty(X);
        internal::check_consistent_length(X, y);

        this->n_features_in_ = X.cols();
        const Eigen::Index n_samples = X.rows();
        const int sample_count = max_samples_.has_value()
            ? std::min<int>(*max_samples_, static_cast<int>(n_samples))
            : static_cast<int>(n_samples);

        // Discover unique class labels (sorted ascending).
        std::vector<int> uniq;
        uniq.reserve(static_cast<std::size_t>(y.size()));
        for (Eigen::Index i = 0; i < y.size(); ++i) uniq.push_back(y(i));
        std::sort(uniq.begin(), uniq.end());
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
        classes_ = Eigen::VectorXi(static_cast<Eigen::Index>(uniq.size()));
        std::map<int, int> class_to_idx;
        for (std::size_t i = 0; i < uniq.size(); ++i) {
            classes_(static_cast<Eigen::Index>(i)) = uniq[i];
            class_to_idx[uniq[i]] = static_cast<int>(i);
        }
        n_classes_val_ = static_cast<int>(uniq.size());

        const uint64_t base_seed = random_state_.value_or(
            static_cast<uint64_t>(std::random_device{}()));

        // Per-tree bootstrap index lists.
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

        // Build fresh estimators (no warm_start).
        estimators_storage_.clear();
        estimators_storage_.reserve(static_cast<std::size_t>(n_estimators_));

        const int tree_max_depth = max_depth_.value_or(-1);
        const int tree_mf_mode = static_cast<int>(max_features_mode_);
        const double tree_mf_value = max_features_value_.has_value()
            ? static_cast<double>(*max_features_value_) : 0.0;

        for (int t = 0; t < n_estimators_; ++t) {
            // Per-tree seed used by the tree's own RNG (feature subspace).
            uint64_t tree_seed = base_seed ^
                (static_cast<uint64_t>(t) * 0x9E3779B97F4A7C15ULL);
            estimators_storage_.emplace_back(
                tree_max_depth, min_samples_split_, tree_mf_mode, tree_mf_value,
                std::optional<uint64_t>(tree_seed));
        }

        // Fit trees (parallel via std::async if n_jobs > 1).
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
                for (auto& f : futs) f.get();  // rethrows any exception.
                next += batch;
            }
        } else {
            for (int t = 0; t < n_estimators_; ++t) fit_one(t);
        }

        // Aggregate feature importances.
        feature_importances_ = RowVectorType::Zero(X.cols());
        for (const auto& tree : estimators_storage_)
            feature_importances_ += tree.feature_importances();
        feature_importances_ /= static_cast<Scalar>(n_estimators_);
        Scalar fi_sum = feature_importances_.sum();
        if (fi_sum > Scalar{0}) feature_importances_ /= fi_sum;

        // OOB scoring.
        if (oob_score_flag_) {
            compute_oob(X, y, per_tree_indices, class_to_idx);
        }

        this->fitted_ = true;
        return *this;
    }

    [[nodiscard]] Eigen::VectorXi predict_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        MatrixType proba = predict_proba(X);
        Eigen::VectorXi out(X.rows());
        for (Eigen::Index i = 0; i < X.rows(); ++i) {
            int best = 0;
            Scalar best_p = proba(i, 0);
            for (int c = 1; c < n_classes_val_; ++c) {
                if (proba(i, c) > best_p) { best_p = proba(i, c); best = c; }
            }
            out(i) = classes_(best);
        }
        return out;
    }

    /// @brief Average of per-tree class probabilities.
    [[nodiscard]] MatrixType predict_proba(
        const Eigen::Ref<const MatrixType>& X) const {
        this->check_is_fitted();
        this->validate_feature_count(X);
        MatrixType acc = MatrixType::Zero(X.rows(), n_classes_val_);
        for (const auto& tree : estimators_storage_) {
            acc += tree.predict_proba(X);
        }
        acc /= static_cast<Scalar>(n_estimators_);
        return acc;
    }

private:
    int n_estimators_;
    CriterionClf criterion_;
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

    std::vector<DecisionTreeClassifier<Scalar>> estimators_storage_;
    Eigen::VectorXi classes_;
    int n_classes_val_ = 0;
    RowVectorType feature_importances_;
    MatrixType oob_decision_function_;
    Scalar oob_score_value_{0};

    void compute_oob(const Eigen::Ref<const MatrixType>& X,
                     const Eigen::Ref<const Eigen::VectorXi>& y,
                     const std::vector<std::vector<Eigen::Index>>& per_tree_indices,
                     const std::map<int, int>& class_to_idx) {
        const Eigen::Index n_samples = X.rows();
        oob_decision_function_ = MatrixType::Zero(n_samples, n_classes_val_);
        Eigen::VectorXi oob_counts = Eigen::VectorXi::Zero(n_samples);

        for (std::size_t t = 0; t < estimators_storage_.size(); ++t) {
            std::vector<bool> in_bag(static_cast<std::size_t>(n_samples), false);
            for (auto idx : per_tree_indices[t])
                in_bag[static_cast<std::size_t>(idx)] = true;

            // Collect OOB row indices.
            std::vector<Eigen::Index> oob_rows;
            for (Eigen::Index i = 0; i < n_samples; ++i)
                if (!in_bag[static_cast<std::size_t>(i)]) oob_rows.push_back(i);

            if (oob_rows.empty()) continue;

            MatrixType X_oob(oob_rows.size(), X.cols());
            for (std::size_t k = 0; k < oob_rows.size(); ++k)
                X_oob.row(static_cast<Eigen::Index>(k)) = X.row(oob_rows[k]);

            MatrixType proba = estimators_storage_[t].predict_proba(X_oob);
            for (std::size_t k = 0; k < oob_rows.size(); ++k) {
                Eigen::Index r = oob_rows[k];
                for (int c = 0; c < n_classes_val_; ++c)
                    oob_decision_function_(r, c) += proba(static_cast<Eigen::Index>(k), c);
                oob_counts(r) += 1;
            }
        }

        int correct = 0;
        int counted = 0;
        for (Eigen::Index i = 0; i < n_samples; ++i) {
            if (oob_counts(i) > 0) {
                Scalar inv = Scalar{1} / static_cast<Scalar>(oob_counts(i));
                int best = 0;
                Scalar best_v = oob_decision_function_(i, 0) * inv;
                oob_decision_function_(i, 0) = best_v;
                for (int c = 1; c < n_classes_val_; ++c) {
                    Scalar v = oob_decision_function_(i, c) * inv;
                    oob_decision_function_(i, c) = v;
                    if (v > best_v) { best_v = v; best = c; }
                }
                int pred_label = classes_(best);
                if (pred_label == y(i)) ++correct;
                ++counted;
            } else {
                for (int c = 0; c < n_classes_val_; ++c)
                    oob_decision_function_(i, c) =
                        std::numeric_limits<Scalar>::quiet_NaN();
            }
        }
        oob_score_value_ = counted > 0
            ? static_cast<Scalar>(correct) / static_cast<Scalar>(counted)
            : Scalar{0};
        (void)class_to_idx;  // currently unused, kept for symmetry.
    }
};

/// @}

} // namespace Skigen

#endif // SKIGEN_ENSEMBLE_RANDOM_FOREST_CLASSIFIER_H
