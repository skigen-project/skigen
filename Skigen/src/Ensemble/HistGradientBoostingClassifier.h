// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_ENSEMBLE_HIST_GRADIENT_BOOSTING_CLASSIFIER_H
#define SKIGEN_ENSEMBLE_HIST_GRADIENT_BOOSTING_CLASSIFIER_H

#include "../Core/Base.h"
#include "../Core/Validation.h"
#include "Detail/HistTree.h"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace Skigen {

/// @addtogroup Algo_RandomForest
/// @{

/// @brief Histogram-based Gradient Boosting for classification.
///
/// Bins each feature into at most `max_bins` quantile-based buckets, then
/// runs stage-wise additive gradient boosting on the binned representation
/// with a native gradient/hessian histogram split finder. Binary problems
/// boost a single log-odds score; multiclass problems boost one tree per
/// class per iteration against the softmax cross-entropy gradient.
///
/// Mirrors
/// [sklearn.ensemble.HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)
/// for the log-loss case.
///
/// ### Limitations relative to scikit-learn
///
/// Split selection uses a native gradient/hessian **histogram** finder
/// with a second-order (Newton) split-gain criterion, best-first
/// **leaf-wise** growth bounded by `max_leaf_nodes`, L2 regularisation,
/// monotonic constraints, and holdout-based early stopping. Both binary
/// and multiclass log-loss are supported. Native categorical features
/// and a native sparse input path remain deferred.
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
        std::optional<std::vector<int>> monotonic_cst = std::nullopt,
        bool early_stopping = false,
        Scalar validation_fraction = Scalar{0.1},
        int n_iter_no_change = 10,
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
          monotonic_cst_(std::move(monotonic_cst)),
          early_stopping_(early_stopping),
          validation_fraction_(validation_fraction),
          n_iter_no_change_(n_iter_no_change),
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

    /// @brief Per-class init scores (length 1 for binary, n_classes otherwise).
    [[nodiscard]] const VectorType& init() const {
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
        return estimators_.empty()
                   ? 0
                   : static_cast<int>(estimators_[0].size());
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
        (validation_fraction, validation_fraction_, double),
        (n_iter_no_change,   n_iter_no_change_,   int),
        (tol,                tol_,                double))

    // -- Fit / Predict ------------------------------------------------------

    HistGradientBoostingClassifier& fit_impl(
        const Eigen::Ref<const MatrixType>& X,
        const Eigen::Ref<const Eigen::VectorXi>& y) {
        internal::check_non_empty(X);
        internal::check_consistent_length(X, y);
        this->n_features_in_ = X.cols();
        const Eigen::Index n = X.rows();

        // Discover the class set.
        std::vector<int> uniq;
        uniq.reserve(static_cast<std::size_t>(y.size()));
        for (Eigen::Index i = 0; i < y.size(); ++i) uniq.push_back(y(i));
        std::sort(uniq.begin(), uniq.end());
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
        if (uniq.size() < 2) {
            throw std::invalid_argument(
                "HistGradientBoostingClassifier: need at least 2 classes; "
                "got " + std::to_string(uniq.size()) + ".");
        }
        const int K = static_cast<int>(uniq.size());
        classes_ = Eigen::VectorXi(K);
        for (int k = 0; k < K; ++k) classes_(k) = uniq[k];

        // Quantile-bin the design matrix.
        using BinnedMatrix = typename internal::HistTree<Scalar>::BinnedMatrix;
        BinnedMatrix X_binned = build_bins(X);

        // Map labels to class indices 0..K-1.
        std::vector<int> y_idx(static_cast<std::size_t>(n));
        for (Eigen::Index i = 0; i < n; ++i) {
            const int* it = std::lower_bound(uniq.data(),
                                             uniq.data() + K, y(i));
            y_idx[static_cast<std::size_t>(i)] =
                static_cast<int>(it - uniq.data());
        }

        // Holdout for early stopping.
        std::vector<int> train_rows, val_rows;
        make_holdout(n, train_rows, val_rows);
        const bool use_es = early_stopping_ && !val_rows.empty();

        internal::HistTreeParams<Scalar> params = make_params();
        n_trees_per_iter_ = (K == 2) ? 1 : K;
        estimators_.assign(static_cast<std::size_t>(n_trees_per_iter_), {});

        std::vector<Scalar> train_scores;
        train_scores.reserve(static_cast<std::size_t>(max_iter_));

        if (n_trees_per_iter_ == 1) {
            // Binary: single log-odds score F.
            VectorType y01(n);
            for (Eigen::Index i = 0; i < n; ++i)
                y01(i) = (y_idx[static_cast<std::size_t>(i)] == 1)
                             ? Scalar{1} : Scalar{0};
            const Scalar p_pos = std::clamp(
                y01.mean(), std::numeric_limits<Scalar>::epsilon(),
                Scalar{1} - std::numeric_limits<Scalar>::epsilon());
            init_ = VectorType::Constant(1, std::log(p_pos /
                                                     (Scalar{1} - p_pos)));
            VectorType F = VectorType::Constant(n, init_(0));

            Scalar best_val = std::numeric_limits<Scalar>::infinity();
            int no_change = 0;
            for (int stage = 0; stage < max_iter_; ++stage) {
                VectorType grad(n), hess(n);
                for (Eigen::Index i = 0; i < n; ++i) {
                    const Scalar pp = sigmoid(F(i));
                    grad(i) = pp - y01(i);
                    hess(i) = std::max(pp * (Scalar{1} - pp),
                                       std::numeric_limits<Scalar>::epsilon());
                }
                if (use_es)
                    for (int v : val_rows) hess(v) = Scalar{0};

                internal::HistTree<Scalar> tree;
                tree.fit(X_binned, grad, hess, params);
                F.noalias() += learning_rate_ * tree.predict(X_binned);
                estimators_[0].push_back(std::move(tree));

                train_scores.push_back(binary_logloss(F, y01));
                if (use_es && early_stop(F, y01, val_rows, best_val,
                                         no_change))
                    break;
            }
        } else {
            // Multiclass: K log-odds scores, one tree per class per stage.
            std::vector<VectorType> F(static_cast<std::size_t>(K),
                                      VectorType::Zero(n));
            init_ = VectorType::Zero(K);
            // Prior log-probabilities as init.
            for (int k = 0; k < K; ++k) {
                Scalar cnt{0};
                for (Eigen::Index i = 0; i < n; ++i)
                    if (y_idx[static_cast<std::size_t>(i)] == k) cnt += 1;
                const Scalar pk = std::clamp(
                    cnt / static_cast<Scalar>(n),
                    std::numeric_limits<Scalar>::epsilon(), Scalar{1});
                init_(k) = std::log(pk);
                F[static_cast<std::size_t>(k)] =
                    VectorType::Constant(n, init_(k));
            }

            Scalar best_val = std::numeric_limits<Scalar>::infinity();
            int no_change = 0;
            for (int stage = 0; stage < max_iter_; ++stage) {
                // Softmax probabilities per sample.
                std::vector<VectorType> P = softmax_rows(F, n, K);
                for (int k = 0; k < K; ++k) {
                    VectorType grad(n), hess(n);
                    for (Eigen::Index i = 0; i < n; ++i) {
                        const Scalar pk = P[static_cast<std::size_t>(k)](i);
                        const Scalar yk =
                            (y_idx[static_cast<std::size_t>(i)] == k)
                                ? Scalar{1} : Scalar{0};
                        grad(i) = pk - yk;
                        hess(i) = std::max(
                            pk * (Scalar{1} - pk),
                            std::numeric_limits<Scalar>::epsilon());
                    }
                    if (use_es)
                        for (int v : val_rows) hess(v) = Scalar{0};
                    internal::HistTree<Scalar> tree;
                    tree.fit(X_binned, grad, hess, params);
                    F[static_cast<std::size_t>(k)].noalias() +=
                        learning_rate_ * tree.predict(X_binned);
                    estimators_[static_cast<std::size_t>(k)].push_back(
                        std::move(tree));
                }
                train_scores.push_back(multiclass_logloss(F, y_idx, n, K));
                if (use_es &&
                    multiclass_early_stop(F, y_idx, val_rows, K, best_val,
                                          no_change))
                    break;
            }
        }

        train_score_ = VectorType(static_cast<Eigen::Index>(
            train_scores.size()));
        for (std::size_t i = 0; i < train_scores.size(); ++i)
            train_score_(static_cast<Eigen::Index>(i)) = train_scores[i];

        this->fitted_ = true;
        return *this;
    }

    [[nodiscard]] LabelType predict_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        this->check_is_fitted();
        const MatrixType P = predict_proba(X);
        LabelType out(X.rows());
        for (Eigen::Index i = 0; i < X.rows(); ++i) {
            Eigen::Index best = 0;
            for (Eigen::Index k = 1; k < P.cols(); ++k)
                if (P(i, k) > P(i, best)) best = k;
            out(i) = classes_(best);
        }
        return out;
    }

    /// @brief Raw additive scores. Shape (n_samples,) for binary (log-odds),
    ///   (n_samples, n_classes) for multiclass.
    [[nodiscard]] MatrixType decision_function(
        const Eigen::Ref<const MatrixType>& X) const {
        this->check_is_fitted();
        const auto X_binned = bin(X);
        const Eigen::Index n = X.rows();
        if (n_trees_per_iter_ == 1) {
            VectorType F = VectorType::Constant(n, init_(0));
            for (const auto& tree : estimators_[0])
                F.noalias() += learning_rate_ * tree.predict(X_binned);
            return F;
        }
        const int K = static_cast<int>(classes_.size());
        MatrixType F(n, K);
        for (int k = 0; k < K; ++k) {
            VectorType fk = VectorType::Constant(n, init_(k));
            for (const auto& tree : estimators_[static_cast<std::size_t>(k)])
                fk.noalias() += learning_rate_ * tree.predict(X_binned);
            F.col(k) = fk;
        }
        return F;
    }

    /// @brief Probability estimates, shape (n_samples, n_classes).
    [[nodiscard]] MatrixType predict_proba(
        const Eigen::Ref<const MatrixType>& X) const {
        const MatrixType D = decision_function(X);
        const Eigen::Index n = X.rows();
        if (n_trees_per_iter_ == 1) {
            MatrixType P(n, 2);
            for (Eigen::Index i = 0; i < n; ++i) {
                const Scalar pp = sigmoid(D(i, 0));
                P(i, 0) = Scalar{1} - pp;
                P(i, 1) = pp;
            }
            return P;
        }
        const int K = static_cast<int>(classes_.size());
        MatrixType P(n, K);
        for (Eigen::Index i = 0; i < n; ++i) {
            Scalar mx = D(i, 0);
            for (int k = 1; k < K; ++k) mx = std::max(mx, D(i, k));
            Scalar denom{0};
            for (int k = 0; k < K; ++k) {
                P(i, k) = std::exp(D(i, k) - mx);
                denom += P(i, k);
            }
            for (int k = 0; k < K; ++k) P(i, k) /= denom;
        }
        return P;
    }

private:
    using BinnedMatrix = typename internal::HistTree<Scalar>::BinnedMatrix;

    static Scalar sigmoid(Scalar x) {
        if (x >= Scalar{0}) {
            const Scalar z = std::exp(-x);
            return Scalar{1} / (Scalar{1} + z);
        }
        const Scalar z = std::exp(x);
        return z / (Scalar{1} + z);
    }

    static std::vector<VectorType> softmax_rows(
        const std::vector<VectorType>& F, Eigen::Index n, int K) {
        std::vector<VectorType> P(static_cast<std::size_t>(K),
                                  VectorType::Zero(n));
        for (Eigen::Index i = 0; i < n; ++i) {
            Scalar mx = F[0](i);
            for (int k = 1; k < K; ++k) mx = std::max(mx, F[static_cast<std::size_t>(k)](i));
            Scalar denom{0};
            std::vector<Scalar> e(static_cast<std::size_t>(K));
            for (int k = 0; k < K; ++k) {
                e[static_cast<std::size_t>(k)] =
                    std::exp(F[static_cast<std::size_t>(k)](i) - mx);
                denom += e[static_cast<std::size_t>(k)];
            }
            for (int k = 0; k < K; ++k)
                P[static_cast<std::size_t>(k)](i) =
                    e[static_cast<std::size_t>(k)] / denom;
        }
        return P;
    }

    static Scalar binary_logloss(const VectorType& F, const VectorType& y01) {
        const Eigen::Index n = F.size();
        Scalar s{0};
        for (Eigen::Index i = 0; i < n; ++i) {
            const Scalar pp = std::clamp(
                sigmoid(F(i)), std::numeric_limits<Scalar>::epsilon(),
                Scalar{1} - std::numeric_limits<Scalar>::epsilon());
            s += -(y01(i) * std::log(pp) +
                   (Scalar{1} - y01(i)) * std::log(Scalar{1} - pp));
        }
        return s / static_cast<Scalar>(n);
    }

    static Scalar multiclass_logloss(const std::vector<VectorType>& F,
                                     const std::vector<int>& y_idx,
                                     Eigen::Index n, int K) {
        const std::vector<VectorType> P = softmax_rows(F, n, K);
        Scalar s{0};
        for (Eigen::Index i = 0; i < n; ++i) {
            const Scalar pk = std::clamp(
                P[static_cast<std::size_t>(y_idx[static_cast<std::size_t>(i)])](i),
                std::numeric_limits<Scalar>::epsilon(), Scalar{1});
            s += -std::log(pk);
        }
        return s / static_cast<Scalar>(n);
    }

    bool early_stop(const VectorType& F, const VectorType& y01,
                    const std::vector<int>& val_rows, Scalar& best_val,
                    int& no_change) const {
        Scalar val{0};
        for (int v : val_rows) {
            const Scalar pp = std::clamp(
                sigmoid(F(v)), std::numeric_limits<Scalar>::epsilon(),
                Scalar{1} - std::numeric_limits<Scalar>::epsilon());
            val += -(y01(v) * std::log(pp) +
                     (Scalar{1} - y01(v)) * std::log(Scalar{1} - pp));
        }
        val /= static_cast<Scalar>(val_rows.size());
        if (val + tol_ < best_val) { best_val = val; no_change = 0; return false; }
        return ++no_change >= n_iter_no_change_;
    }

    bool multiclass_early_stop(const std::vector<VectorType>& F,
                               const std::vector<int>& y_idx,
                               const std::vector<int>& val_rows, int K,
                               Scalar& best_val, int& no_change) const {
        const Eigen::Index n = F[0].size();
        const std::vector<VectorType> P = softmax_rows(F, n, K);
        Scalar val{0};
        for (int v : val_rows) {
            const Scalar pk = std::clamp(
                P[static_cast<std::size_t>(y_idx[static_cast<std::size_t>(v)])](v),
                std::numeric_limits<Scalar>::epsilon(), Scalar{1});
            val += -std::log(pk);
        }
        val /= static_cast<Scalar>(val_rows.size());
        if (val + tol_ < best_val) { best_val = val; no_change = 0; return false; }
        return ++no_change >= n_iter_no_change_;
    }

    BinnedMatrix build_bins(const Eigen::Ref<const MatrixType>& X) {
        const Eigen::Index n = X.rows();
        const Eigen::Index p = X.cols();
        bin_edges_.assign(static_cast<std::size_t>(p), {});
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
                const Scalar t = sorted_col[idx];
                if (thresholds.empty() || t > thresholds.back())
                    thresholds.push_back(t);
            }
            bin_edges_[static_cast<std::size_t>(j)] = thresholds;
        }
        return bin(X);
    }

    [[nodiscard]] BinnedMatrix bin(
        const Eigen::Ref<const MatrixType>& X) const {
        const Eigen::Index n = X.rows();
        const Eigen::Index p = X.cols();
        BinnedMatrix Xb(n, p);
        for (Eigen::Index j = 0; j < p; ++j) {
            const auto& thr = bin_edges_[static_cast<std::size_t>(j)];
            for (Eigen::Index i = 0; i < n; ++i) {
                auto it = std::upper_bound(thr.begin(), thr.end(), X(i, j));
                Xb(i, j) = static_cast<uint8_t>(
                    std::distance(thr.begin(), it));
            }
        }
        return Xb;
    }

    internal::HistTreeParams<Scalar> make_params() const {
        internal::HistTreeParams<Scalar> params;
        params.max_leaf_nodes = max_leaf_nodes_.value_or(0);
        params.max_depth = max_depth_.value_or(-1);
        params.min_samples_leaf = min_samples_leaf_;
        params.l2_regularization = l2_regularization_;
        params.n_bins = max_bins_;
        if (monotonic_cst_.has_value()) params.monotonic_cst = *monotonic_cst_;
        return params;
    }

    void make_holdout(Eigen::Index n, std::vector<int>& train_rows,
                      std::vector<int>& val_rows) const {
        train_rows.clear();
        val_rows.clear();
        if (!early_stopping_ || validation_fraction_ <= Scalar{0}) return;
        std::vector<int> perm(static_cast<std::size_t>(n));
        for (Eigen::Index i = 0; i < n; ++i)
            perm[static_cast<std::size_t>(i)] = static_cast<int>(i);
        std::mt19937_64 rng(random_state_.value_or(0ULL));
        std::shuffle(perm.begin(), perm.end(), rng);
        const auto n_val = static_cast<std::size_t>(
            static_cast<double>(n) * validation_fraction_);
        if (n_val == 0 || n_val >= perm.size()) return;
        val_rows.assign(perm.begin(),
                        perm.begin() + static_cast<std::ptrdiff_t>(n_val));
        train_rows.assign(perm.begin() + static_cast<std::ptrdiff_t>(n_val),
                          perm.end());
    }

    Loss loss_;
    Scalar learning_rate_;
    int max_iter_;
    std::optional<int> max_leaf_nodes_;
    std::optional<int> max_depth_;
    int min_samples_leaf_;
    Scalar l2_regularization_;
    int max_bins_;
    std::optional<std::vector<int>> monotonic_cst_;
    bool early_stopping_;
    Scalar validation_fraction_;
    int n_iter_no_change_;
    Scalar tol_;
    std::optional<uint64_t> random_state_;

    int n_trees_per_iter_ = 1;
    VectorType init_;
    Eigen::VectorXi classes_;
    std::vector<std::vector<Scalar>> bin_edges_;
    // estimators_[k] is the additive sequence of trees for class k
    // (binary uses a single sequence in estimators_[0]).
    std::vector<std::vector<internal::HistTree<Scalar>>> estimators_;
    VectorType train_score_;
};

/// @}

} // namespace Skigen

#endif // SKIGEN_ENSEMBLE_HIST_GRADIENT_BOOSTING_CLASSIFIER_H
