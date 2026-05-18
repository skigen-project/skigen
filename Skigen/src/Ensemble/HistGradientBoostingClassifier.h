// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_ENSEMBLE_HIST_GRADIENT_BOOSTING_CLASSIFIER_H
#define SKIGEN_ENSEMBLE_HIST_GRADIENT_BOOSTING_CLASSIFIER_H

#include "../Core/Base.h"
#include "../Core/Validation.h"
#include "../Tree/DecisionTree.h"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace Skigen {

/// @addtogroup Algo_RandomForest
/// @{

/// @brief Histogram-based Gradient Boosting for binary classification.
///
/// Bins each feature into at most `max_bins` quantile-based buckets, then
/// runs stage-wise additive log-odds gradient boosting (the same scheme
/// as `GradientBoostingClassifier`) on the binned representation.
///
/// Mirrors
/// [sklearn.ensemble.HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)
/// for the binary log-loss case.
///
/// ### Limitations relative to scikit-learn
///
/// Binary classification only; multiclass is rejected at fit time.
/// Once X is binned, the existing `DecisionTreeRegressor` is used for
/// split selection rather than a native histogram-based split finder
/// — predictions match when the binning is fine enough but the
/// scaling-on-large-n advantage of histograms is not realised.
/// Leaf-wise growth (`max_leaf_nodes`), monotonic constraints, native
/// categoricals, early stopping, and `l2_regularization` are accepted
/// as constructor parameters but are not honoured at fit time.
template <typename Scalar = double>
class HistGradientBoostingClassifier
    : public Classifier<HistGradientBoostingClassifier<Scalar>, Scalar> {
public:
    using Base = Classifier<HistGradientBoostingClassifier<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;
    using typename Base::ScalarType;
    using typename Base::LabelType;
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    enum class Loss { LogLoss };

    explicit HistGradientBoostingClassifier(
        Loss loss = Loss::LogLoss,
        Scalar learning_rate = Scalar{0.1},
        int max_iter = 100,
        std::optional<int> max_leaf_nodes = 31,
        std::optional<int> max_depth = std::nullopt,
        int min_samples_leaf = 20,
        Scalar l2_regularization = Scalar{0},
        int max_bins = 255,
        bool early_stopping = false,
        Scalar tol = Scalar{1e-7},
        std::optional<uint64_t> random_state = std::nullopt)
        : loss_(loss),
          learning_rate_(learning_rate),
          max_iter_(max_iter),
          max_leaf_nodes_(max_leaf_nodes),
          max_depth_(max_depth),
          min_samples_leaf_(min_samples_leaf),
          l2_regularization_(l2_regularization),
          max_bins_(max_bins),
          early_stopping_(early_stopping),
          tol_(tol),
          random_state_(random_state) {
        if (max_bins_ < 2 || max_bins_ > 255) {
            throw std::invalid_argument(
                "max_bins must be in [2, 255]; got " +
                std::to_string(max_bins_));
        }
    }

    // -- Accessors ----------------------------------------------------------

    [[nodiscard]] Loss loss() const noexcept { return loss_; }
    [[nodiscard]] Scalar learning_rate() const noexcept {
        return learning_rate_;
    }
    [[nodiscard]] int max_iter() const noexcept { return max_iter_; }
    [[nodiscard]] int max_bins() const noexcept { return max_bins_; }

    [[nodiscard]] Scalar init() const {
        this->check_is_fitted(); return init_;
    }
    [[nodiscard]] const Eigen::VectorXi& classes() const {
        this->check_is_fitted(); return classes_;
    }
    [[nodiscard]] int n_classes() const {
        this->check_is_fitted(); return static_cast<int>(classes_.size());
    }
    [[nodiscard]] int n_iter() const {
        this->check_is_fitted();
        return static_cast<int>(estimators_.size());
    }
    [[nodiscard]] const std::vector<std::vector<Scalar>>& bin_edges() const {
        this->check_is_fitted(); return bin_edges_;
    }
    [[nodiscard]] const VectorType& train_score() const {
        this->check_is_fitted(); return train_score_;
    }

    SKIGEN_PARAMS(
        (learning_rate,      learning_rate_,      double),
        (max_iter,           max_iter_,           int),
        (min_samples_leaf,   min_samples_leaf_,   int),
        (l2_regularization,  l2_regularization_,  double),
        (max_bins,           max_bins_,           int),
        (early_stopping,     early_stopping_,     bool),
        (tol,                tol_,                double))

    // -- Fit / Predict ------------------------------------------------------

    HistGradientBoostingClassifier& fit_impl(
        const Eigen::Ref<const MatrixType>& X,
        const Eigen::Ref<const Eigen::VectorXi>& y) {
        internal::check_non_empty(X);
        internal::check_consistent_length(X, y);
        this->n_features_in_ = X.cols();
        const Eigen::Index n = X.rows();
        const Eigen::Index p = X.cols();

        // Discover & validate the class set (binary only).
        std::vector<int> uniq;
        uniq.reserve(static_cast<std::size_t>(y.size()));
        for (Eigen::Index i = 0; i < y.size(); ++i) uniq.push_back(y(i));
        std::sort(uniq.begin(), uniq.end());
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
        if (uniq.size() != 2) {
            throw std::invalid_argument(
                "HistGradientBoostingClassifier: only "
                "binary classification (n_classes=2) is supported; got " +
                std::to_string(uniq.size()) + " classes.");
        }
        classes_ = Eigen::VectorXi(2);
        classes_(0) = uniq[0];
        classes_(1) = uniq[1];

        // 1. Quantile-based binning per feature.
        bin_edges_.assign(static_cast<std::size_t>(p), {});
        MatrixType X_binned(n, p);
        for (Eigen::Index j = 0; j < p; ++j) {
            std::vector<Scalar> sorted_col(static_cast<std::size_t>(n));
            for (Eigen::Index i = 0; i < n; ++i) sorted_col[i] = X(i, j);
            std::sort(sorted_col.begin(), sorted_col.end());

            std::vector<Scalar> thresholds;
            const int n_thresh = max_bins_ - 1;
            for (int b = 1; b <= n_thresh; ++b) {
                const std::size_t idx = std::min(
                    static_cast<std::size_t>(
                        static_cast<double>(b) * static_cast<double>(n) /
                        static_cast<double>(max_bins_)),
                    sorted_col.size() - 1);
                Scalar t = sorted_col[idx];
                if (thresholds.empty() || t > thresholds.back()) {
                    thresholds.push_back(t);
                }
            }
            bin_edges_[static_cast<std::size_t>(j)] = thresholds;

            for (Eigen::Index i = 0; i < n; ++i) {
                auto it = std::upper_bound(
                    thresholds.begin(), thresholds.end(), X(i, j));
                X_binned(i, j) = static_cast<Scalar>(
                    std::distance(thresholds.begin(), it));
            }
        }

        // 2. Encode y as 0 / 1 against classes_(1) being positive.
        const int pos_label = uniq[1];
        VectorType y01(n);
        for (Eigen::Index i = 0; i < n; ++i) {
            y01(i) = (y(i) == pos_label) ? Scalar{1} : Scalar{0};
        }

        // 3. Initial log-odds = log(p_+/p_-) (matches sklearn's init).
        const Scalar p_pos = std::clamp(
            y01.mean(),
            std::numeric_limits<Scalar>::epsilon(),
            Scalar{1} - std::numeric_limits<Scalar>::epsilon());
        init_ = std::log(p_pos / (Scalar{1} - p_pos));

        VectorType F = VectorType::Constant(n, init_);
        estimators_.clear();
        estimators_.reserve(static_cast<std::size_t>(max_iter_));
        train_score_ = VectorType::Zero(max_iter_);

        const uint64_t base_seed = random_state_.value_or(0ULL);
        const int max_depth_eff = max_depth_.value_or(-1);

        for (int stage = 0; stage < max_iter_; ++stage) {
            VectorType residuals(n);
            for (Eigen::Index i = 0; i < n; ++i) {
                residuals(i) = y01(i) - sigmoid(F(i));
            }
            DecisionTreeRegressor<Scalar> tree(
                max_depth_eff,
                min_samples_leaf_ * 2,
                /*max_features_mode=*/0,
                /*max_features_value=*/0.0,
                random_state_.has_value()
                    ? std::optional<uint64_t>(
                          base_seed ^ static_cast<uint64_t>(stage))
                    : std::nullopt);
            tree.fit(X_binned, residuals);

            VectorType update = tree.predict(X_binned);
            F.noalias() += learning_rate_ * update;

            Scalar loss_sum{0};
            for (Eigen::Index i = 0; i < n; ++i) {
                const Scalar pp = sigmoid(F(i));
                const Scalar safe = std::clamp(
                    pp, std::numeric_limits<Scalar>::epsilon(),
                    Scalar{1} - std::numeric_limits<Scalar>::epsilon());
                loss_sum += -(y01(i) * std::log(safe) +
                              (Scalar{1} - y01(i)) *
                              std::log(Scalar{1} - safe));
            }
            train_score_(stage) = loss_sum / static_cast<Scalar>(n);
            estimators_.push_back(std::move(tree));
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

    /// @brief Raw additive score (log-odds), shape (n_samples,).
    [[nodiscard]] VectorType decision_function(
        const Eigen::Ref<const MatrixType>& X) const {
        this->check_is_fitted();
        MatrixType X_binned = bin(X);
        VectorType F = VectorType::Constant(X.rows(), init_);
        for (const auto& tree : estimators_) {
            F.noalias() += learning_rate_ * tree.predict(X_binned);
        }
        return F;
    }

    /// @brief Probability estimates, shape (n_samples, 2).
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

    MatrixType bin(const Eigen::Ref<const MatrixType>& X) const {
        const Eigen::Index n = X.rows();
        const Eigen::Index p = X.cols();
        MatrixType Xb(n, p);
        for (Eigen::Index j = 0; j < p; ++j) {
            const auto& thresholds =
                bin_edges_[static_cast<std::size_t>(j)];
            for (Eigen::Index i = 0; i < n; ++i) {
                auto it = std::upper_bound(
                    thresholds.begin(), thresholds.end(), X(i, j));
                Xb(i, j) = static_cast<Scalar>(
                    std::distance(thresholds.begin(), it));
            }
        }
        return Xb;
    }

    Loss loss_;
    Scalar learning_rate_;
    int max_iter_;
    std::optional<int> max_leaf_nodes_;
    std::optional<int> max_depth_;
    int min_samples_leaf_;
    Scalar l2_regularization_;
    int max_bins_;
    bool early_stopping_;
    Scalar tol_;
    std::optional<uint64_t> random_state_;

    Scalar init_{0};
    Eigen::VectorXi classes_;
    std::vector<std::vector<Scalar>> bin_edges_;
    std::vector<DecisionTreeRegressor<Scalar>> estimators_;
    VectorType train_score_;
};

/// @}

} // namespace Skigen

#endif // SKIGEN_ENSEMBLE_HIST_GRADIENT_BOOSTING_CLASSIFIER_H
