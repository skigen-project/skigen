// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_ENSEMBLE_GRADIENT_BOOSTING_CLASSIFIER_H
#define SKIGEN_ENSEMBLE_GRADIENT_BOOSTING_CLASSIFIER_H

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

/// @brief Gradient Boosting for binary classification.
///
/// Stage-wise additive log-odds model fit by gradient boosting on the
/// negative log-likelihood (cross-entropy) of a binomial outcome:
///
/// @f[
///   F_M(x) = F_0 + \eta \sum_{m=1}^M h_m(x), \qquad
///   P(y=1 \mid x) = \sigma(F_M(x)).
/// @f]
///
/// At each stage the pseudo-residual is the negative gradient of the loss:
/// @f$ r_i = y_i - \sigma(F_{m-1}(x_i)) @f$, where @f$ y_i \in \{0, 1\} @f$.
/// The initial log-odds @f$ F_0 @f$ is the empirical prior log-ratio
/// @f$ \log(p_+ / p_-) @f$, matching sklearn's `init="zero"` default which
/// uses a `DummyClassifier(strategy="prior")` under the hood.
///
/// Mirrors
/// [sklearn.ensemble.GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
/// for the binary case.
///
/// ### Parameters
///
/// Same constructor signature as sklearn's. Defaults: `loss=LogLoss`,
/// `learning_rate=0.1`, `n_estimators=100`, `subsample=1.0`,
/// `criterion=FriedmanMSE`, `min_samples_split=2`, `min_samples_leaf=1`,
/// `max_depth=3`, `validation_fraction=0.1`, `tol=1e-4`. Multiclass,
/// stochastic boosting, and early stopping are not implemented.
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type |
/// |---|---|
/// | `estimators()` | `vector<DecisionTreeRegressor>` |
/// | `classes()` | `Eigen::VectorXi` (length 2) |
/// | `init()` | `Scalar` (initial log-odds) |
/// | `feature_importances()` | `RowVectorType` |
/// | `train_score()` | `VectorType` (per-stage log-loss) |
template <typename Scalar = double>
class GradientBoostingClassifier
    : public Classifier<GradientBoostingClassifier<Scalar>, Scalar> {
public:
    using Base = Classifier<GradientBoostingClassifier<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;
    using typename Base::ScalarType;
    using typename Base::LabelType;
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    enum class Loss { LogLoss, Exponential };
    enum class CriterionGB { FriedmanMSE, SquaredError };

    explicit GradientBoostingClassifier(
        Loss loss = Loss::LogLoss,
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
          verbose_(verbose),
          max_leaf_nodes_(max_leaf_nodes),
          warm_start_(warm_start),
          validation_fraction_(validation_fraction),
          n_iter_no_change_(n_iter_no_change),
          tol_(tol),
          ccp_alpha_(ccp_alpha) {
        if (loss_ != Loss::LogLoss) {
            throw std::invalid_argument(
                "GradientBoostingClassifier: only loss=LogLoss is implemented "
                "in Exponential is not honoured.");
        }
        if (subsample_ <= Scalar{0} || subsample_ > Scalar{1}) {
            throw std::invalid_argument(
                "subsample must be in (0, 1]; got " +
                std::to_string(static_cast<double>(subsample_)));
        }
        if (subsample_ < Scalar{1}) {
            throw std::invalid_argument(
                "GradientBoostingClassifier: subsample < 1.0 (stochastic GB) "
                "is not implemented.");
        }
    }

    // -- Accessors ----------------------------------------------------------

    [[nodiscard]] Loss loss() const noexcept { return loss_; }
    [[nodiscard]] Scalar learning_rate() const noexcept { return learning_rate_; }
    [[nodiscard]] int n_estimators() const noexcept { return n_estimators_; }
    [[nodiscard]] int max_depth() const noexcept { return max_depth_; }

    [[nodiscard]] Scalar init() const {
        this->check_is_fitted(); return init_;
    }
    [[nodiscard]] const Eigen::VectorXi& classes() const {
        this->check_is_fitted(); return classes_;
    }
    [[nodiscard]] int n_classes() const {
        this->check_is_fitted(); return static_cast<int>(classes_.size());
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
        (verbose,                  verbose_,                  int),
        (warm_start,               warm_start_,               bool),
        (validation_fraction,      validation_fraction_,       double),
        (tol,                      tol_,                      double),
        (ccp_alpha,                ccp_alpha_,                double))

    // -- Fit/Predict --------------------------------------------------------

    GradientBoostingClassifier& fit_impl(
        const Eigen::Ref<const MatrixType>& X,
        const Eigen::Ref<const Eigen::VectorXi>& y) {
        internal::check_non_empty(X);
        internal::check_consistent_length(X, y);

        this->n_features_in_ = X.cols();
        const Eigen::Index n = X.rows();

        // Discover unique class labels (sorted ascending).
        std::vector<int> uniq;
        uniq.reserve(static_cast<std::size_t>(y.size()));
        for (Eigen::Index i = 0; i < y.size(); ++i) uniq.push_back(y(i));
        std::sort(uniq.begin(), uniq.end());
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());

        if (uniq.size() != 2) {
            throw std::invalid_argument(
                "GradientBoostingClassifier: only binary "
                "classification (n_classes=2) is supported; got " +
                std::to_string(uniq.size()) + " classes.");
        }
        classes_ = Eigen::VectorXi(2);
        classes_(0) = uniq[0];
        classes_(1) = uniq[1];

        // Encode targets as 0/1 against classes_(1) being the positive class.
        VectorType y01(n);
        const int pos_label = uniq[1];
        for (Eigen::Index i = 0; i < n; ++i) {
            y01(i) = (y(i) == pos_label) ? Scalar{1} : Scalar{0};
        }

        // Initial log-odds = log(p_+ / p_-). Clamp p_+ ∈ [eps, 1-eps] to
        // avoid singular F_0 when one class is missing in tiny test datasets.
        const Scalar p_pos = std::clamp(
            y01.mean(),
            std::numeric_limits<Scalar>::epsilon(),
            Scalar{1} - std::numeric_limits<Scalar>::epsilon());
        init_ = std::log(p_pos / (Scalar{1} - p_pos));

        VectorType F = VectorType::Constant(n, init_);
        estimators_storage_.clear();
        estimators_storage_.reserve(static_cast<std::size_t>(n_estimators_));

        feature_importances_ = RowVectorType::Zero(X.cols());
        train_score_ = VectorType::Zero(n_estimators_);

        const uint64_t base_seed = random_state_.value_or(0ULL);

        for (int stage = 0; stage < n_estimators_; ++stage) {
            // Pseudo-residuals for log-loss: y - sigmoid(F).
            VectorType residuals(n);
            for (Eigen::Index i = 0; i < n; ++i) {
                residuals(i) = y01(i) - sigmoid(F(i));
            }

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
            F.noalias() += learning_rate_ * update;

            // Per-stage training log-loss (mean over samples).
            Scalar loss_sum{0};
            for (Eigen::Index i = 0; i < n; ++i) {
                const Scalar p = sigmoid(F(i));
                const Scalar safe = std::clamp(
                    p, std::numeric_limits<Scalar>::epsilon(),
                    Scalar{1} - std::numeric_limits<Scalar>::epsilon());
                loss_sum += -(y01(i) * std::log(safe) +
                              (Scalar{1} - y01(i)) * std::log(Scalar{1} - safe));
            }
            train_score_(stage) = loss_sum / static_cast<Scalar>(n);

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

    [[nodiscard]] LabelType predict_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        VectorType F = decision_function(X);
        LabelType out(X.rows());
        for (Eigen::Index i = 0; i < X.rows(); ++i) {
            out(i) = (F(i) >= Scalar{0}) ? classes_(1) : classes_(0);
        }
        return out;
    }

    /// @brief Raw additive score @f$ F(x) @f$ (log-odds). Length n_samples.
    [[nodiscard]] VectorType decision_function(
        const Eigen::Ref<const MatrixType>& X) const {
        this->check_is_fitted();
        VectorType F = VectorType::Constant(X.rows(), init_);
        for (const auto& tree : estimators_storage_) {
            F.noalias() += learning_rate_ * tree.predict(X);
        }
        return F;
    }

    /// @brief Probability estimates of shape (n_samples, 2). Column ordering
    ///   matches `classes()`.
    [[nodiscard]] MatrixType predict_proba(
        const Eigen::Ref<const MatrixType>& X) const {
        VectorType F = decision_function(X);
        MatrixType P(X.rows(), 2);
        for (Eigen::Index i = 0; i < X.rows(); ++i) {
            const Scalar p_pos = sigmoid(F(i));
            P(i, 0) = Scalar{1} - p_pos;
            P(i, 1) = p_pos;
        }
        return P;
    }

private:
    static Scalar sigmoid(Scalar x) {
        if (x >= Scalar{0}) {
            const Scalar z = std::exp(-x);
            return Scalar{1} / (Scalar{1} + z);
        }
        const Scalar z = std::exp(x);
        return z / (Scalar{1} + z);
    }

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
    int verbose_;
    std::optional<int> max_leaf_nodes_;
    bool warm_start_;
    Scalar validation_fraction_;
    std::optional<int> n_iter_no_change_;
    Scalar tol_;
    Scalar ccp_alpha_;

    Scalar init_{0};
    Eigen::VectorXi classes_;
    std::vector<DecisionTreeRegressor<Scalar>> estimators_storage_;
    RowVectorType feature_importances_;
    VectorType train_score_;
};

/// @}

} // namespace Skigen

#endif // SKIGEN_ENSEMBLE_GRADIENT_BOOSTING_CLASSIFIER_H
