// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_ENSEMBLE_HIST_GRADIENT_BOOSTING_REGRESSOR_H
#define SKIGEN_ENSEMBLE_HIST_GRADIENT_BOOSTING_REGRESSOR_H

#include "../Core/Base.h"
#include "../Core/Validation.h"
#include "Detail/HistTree.h"

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <algorithm>
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

/// @brief Histogram-based Gradient Boosting for regression.
///
/// Bins each feature into at most `max_bins` quantile-based buckets up
/// front, then runs stage-wise additive gradient boosting on the binned
/// representation. The binning step is what gives sklearn's
/// `HistGradientBoostingRegressor` its scaling advantage on large
/// datasets — split candidates collapse from `n_samples` distinct
/// thresholds per feature down to `max_bins`.
///
/// Mirrors
/// [sklearn.ensemble.HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)
/// for the squared-error loss case.
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default |
/// |---|---|---|
/// | `loss` | `Loss` | `SquaredError` |
/// | `learning_rate` | `Scalar` | `0.1` |
/// | `max_iter` | `int` | `100` |
/// | `max_leaf_nodes` | `optional<int>` | `31` *(leaf-wise growth bound; `nullopt` = unbounded)* |
/// | `max_depth` | `optional<int>` | `nullopt` *(unlimited)* |
/// | `min_samples_leaf` | `int` | `20` |
/// | `l2_regularization` | `Scalar` | `0.0` *(Newton-step shrinkage)* |
/// | `max_bins` | `int` | `255` |
/// | `categorical_features` | `optional<vector<int>>` | `nullopt` *(ignored — treated as ordinals)* |
/// | `monotonic_cst` | `optional<vector<int>>` | `nullopt` *(per-feature +1 / -1 / 0)* |
/// | `early_stopping` | `bool` | `false` *(holdout-based stopping on `validation_fraction`)* |
/// | `validation_fraction` | `Scalar` | `0.1` *(holdout size for early stopping)* |
/// | `n_iter_no_change` | `int` | `10` *(patience for early stopping)* |
/// | `tol` | `Scalar` | `1e-7` *(min validation improvement)* |
/// | `random_state` | `optional<uint64_t>` | `nullopt` |
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type |
/// |---|---|
/// | `init()` | `Scalar` (mean of y) |
/// | `n_iter()` | `int` (== max_iter; no early-stopping yet) |
/// | `bin_edges()` | `vector<vector<Scalar>>` (per-feature thresholds) |
/// | `train_score()` | `VectorType` (per-stage MSE) |
///
/// ### Limitations relative to scikit-learn
///
/// Only `loss=SquaredError` is supported; AbsoluteError, Poisson, and
/// Quantile losses raise on construction. Split selection uses a native
/// gradient/hessian **histogram** finder with a second-order (Newton)
/// split-gain criterion, best-first **leaf-wise** growth bounded by
/// `max_leaf_nodes`, L2 regularisation, monotonic constraints, and
/// holdout-based early stopping. **Native categorical features** are
/// supported: features listed in `categorical_features` are integer-coded,
/// binned by identity, and split by the unordered gradient-sorted strategy
/// (LightGBM / sklearn). A native sparse input path remains deferred.
template <typename Scalar = double>
class HistGradientBoostingRegressor
    : public Predictor<HistGradientBoostingRegressor<Scalar>, Scalar> {
public:
    using Base = Predictor<HistGradientBoostingRegressor<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;
    using typename Base::ScalarType;
    using BinnedMatrix = typename internal::HistTree<Scalar>::BinnedMatrix;

    // Make the dense base-class fit/predict overloads visible alongside the
    // sparse overloads added below.
    using Base::fit;
    using Base::predict;

    enum class Loss { SquaredError, AbsoluteError, Poisson, Quantile };

    explicit HistGradientBoostingRegressor(
        Loss loss = Loss::SquaredError,
        Scalar learning_rate = Scalar{0.1},
        int max_iter = 100,
        std::optional<int> max_leaf_nodes = 31,
        std::optional<int> max_depth = std::nullopt,
        int min_samples_leaf = 20,
        Scalar l2_regularization = Scalar{0},
        int max_bins = 255,
        std::optional<std::vector<int>> categorical_features = std::nullopt,
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
          categorical_features_(std::move(categorical_features)),
          monotonic_cst_(std::move(monotonic_cst)),
          early_stopping_(early_stopping),
          validation_fraction_(validation_fraction),
          n_iter_no_change_(n_iter_no_change),
          tol_(tol),
          random_state_(random_state) {
        if (loss_ != Loss::SquaredError) {
            throw std::invalid_argument(
                "HistGradientBoostingRegressor: only loss=SquaredError is "
                "implemented.");
        }
        if (max_bins_ < 2 || max_bins_ > 255) {
            throw std::invalid_argument(
                "max_bins must be in [2, 255]; got " +
                std::to_string(max_bins_));
        }
    }

    // -- Accessors ----------------------------------------------------------

    [[nodiscard]] Loss   loss()          const noexcept { return loss_; }
    [[nodiscard]] Scalar learning_rate() const noexcept { return learning_rate_; }
    [[nodiscard]] int    max_iter()      const noexcept { return max_iter_; }
    [[nodiscard]] int    max_bins()      const noexcept { return max_bins_; }

    [[nodiscard]] Scalar init() const {
        this->check_is_fitted(); return init_;
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
        (validation_fraction, validation_fraction_, double),
        (n_iter_no_change,   n_iter_no_change_,   int),
        (tol,                tol_,                double))

    // -- Fit / Predict ------------------------------------------------------

    HistGradientBoostingRegressor& fit_impl(
        const Eigen::Ref<const MatrixType>& X,
        const Eigen::Ref<const VectorType>& y) {
        internal::check_non_empty(X);
        internal::check_consistent_length(X, y);
        this->n_features_in_ = X.cols();
        // Quantile-bin each feature into a compact uint8 representation,
        // then run the shared binned-boosting core.
        BinnedMatrix X_binned = build_bins(X);
        fit_from_binned(X_binned, y);
        return *this;
    }

    /// @brief Fit natively from a sparse design matrix.
    ///
    /// The sparse input is binned directly into the compact uint8 histogram
    /// representation without ever materialising a dense `double` @f$n\times
    /// p@f$ matrix (sklearn's HistGB likewise always operates on a dense
    /// `uint8` binned matrix). Implicit zeros are binned as the value-0 bin.
    template <int Options, typename StorageIndex>
    HistGradientBoostingRegressor& fit(
        const Eigen::SparseMatrix<Scalar, Options, StorageIndex>& X,
        const Eigen::Ref<const VectorType>& y) {
        if (X.rows() == 0 || X.cols() == 0)
            throw std::invalid_argument(
                "HistGradientBoostingRegressor.fit: empty sparse matrix.");
        if (X.rows() != y.size())
            throw std::invalid_argument(
                "HistGradientBoostingRegressor.fit: X has " +
                std::to_string(X.rows()) + " rows but y has " +
                std::to_string(y.size()) + ".");
        this->n_features_in_ = X.cols();
        BinnedMatrix X_binned = build_bins_sparse(X);
        fit_from_binned(X_binned, y);
        return *this;
    }

    // Shared boosting core operating on the pre-binned uint8 matrix.
    void fit_from_binned(const BinnedMatrix& X_binned,
                         const Eigen::Ref<const VectorType>& y) {
        const Eigen::Index n = X_binned.rows();

        // Optional holdout split for early stopping.
        std::vector<int> train_rows, val_rows;
        make_holdout(n, train_rows, val_rows);
        const bool use_es = early_stopping_ && !val_rows.empty();

        // Initialise predictions at the marginal mean (matches sklearn's
        // DummyRegressor(strategy="mean") default init).
        init_ = y.mean();
        VectorType F = VectorType::Constant(n, init_);

        estimators_.clear();
        estimators_.reserve(static_cast<std::size_t>(max_iter_));
        std::vector<Scalar> train_scores;
        train_scores.reserve(static_cast<std::size_t>(max_iter_));

        internal::HistTreeParams<Scalar> params = make_params();

        Scalar best_val = std::numeric_limits<Scalar>::infinity();
        int no_change = 0;

        for (int stage = 0; stage < max_iter_; ++stage) {
            // Squared-error: gradient = F - y, hessian = 1.
            VectorType grad(n), hess(n);
            for (Eigen::Index i = 0; i < n; ++i) {
                grad(i) = F(i) - y(i);
                hess(i) = Scalar{1};
            }

            internal::HistTree<Scalar> tree;
            if (use_es) {
                // Fit only on the training rows by zeroing the holdout
                // hessian so those samples carry no weight in any split.
                VectorType hess_tr = hess;
                for (int v : val_rows) hess_tr(v) = Scalar{0};
                tree.fit(X_binned, grad, hess_tr, params);
            } else {
                tree.fit(X_binned, grad, hess, params);
            }

            const VectorType update = tree.predict(X_binned);
            F.noalias() += learning_rate_ * update;

            train_scores.push_back(
                ((y - F).array().square().sum()) / static_cast<Scalar>(n));
            estimators_.push_back(std::move(tree));

            if (use_es) {
                Scalar val_err{0};
                for (int v : val_rows) {
                    const Scalar r = y(v) - F(v);
                    val_err += r * r;
                }
                val_err /= static_cast<Scalar>(val_rows.size());
                if (val_err + tol_ < best_val) {
                    best_val = val_err;
                    no_change = 0;
                } else if (++no_change >= n_iter_no_change_) {
                    break;
                }
            }
        }

        train_score_ = VectorType(static_cast<Eigen::Index>(
            train_scores.size()));
        for (std::size_t i = 0; i < train_scores.size(); ++i)
            train_score_(static_cast<Eigen::Index>(i)) = train_scores[i];

        this->fitted_ = true;
    }

    [[nodiscard]] VectorType predict_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        const Eigen::Index n = X.rows();
        const auto X_binned = bin(X);
        VectorType F = VectorType::Constant(n, init_);
        for (const auto& tree : estimators_) {
            F.noalias() += learning_rate_ * tree.predict(X_binned);
        }
        return F;
    }

    /// @brief Predict from a sparse design matrix without densifying.
    template <int Options, typename StorageIndex>
    [[nodiscard]] VectorType predict(
        const Eigen::SparseMatrix<Scalar, Options, StorageIndex>& X) const {
        this->check_is_fitted();
        const Eigen::Index n = X.rows();
        const auto X_binned = bin_sparse(X);
        VectorType F = VectorType::Constant(n, init_);
        for (const auto& tree : estimators_) {
            F.noalias() += learning_rate_ * tree.predict(X_binned);
        }
        return F;
    }

    [[nodiscard]] Scalar score_impl(
        const Eigen::Ref<const MatrixType>& X,
        const Eigen::Ref<const VectorType>& y) const {
        VectorType yhat = predict_impl(X);
        const Scalar ym = y.mean();
        Scalar ss_res{0}, ss_tot{0};
        for (Eigen::Index i = 0; i < y.size(); ++i) {
            const Scalar r = y(i) - yhat(i);
            ss_res += r * r;
            const Scalar d = y(i) - ym;
            ss_tot += d * d;
        }
        if (ss_tot == Scalar{0}) return Scalar{0};
        return Scalar{1} - ss_res / ss_tot;
    }

private:
    // Build per-feature quantile thresholds and the binned matrix.
    // Categorical features (listed in `categorical_features_`) are binned by
    // identity: integer-coded category `k` maps to bin `k` via threshold
    // edges at the half-integers `0.5, 1.5, ...`. Numerical features use
    // quantile edges as before.
    BinnedMatrix build_bins(const Eigen::Ref<const MatrixType>& X) {
        const Eigen::Index n = X.rows();
        const Eigen::Index p = X.cols();
        bin_edges_.assign(static_cast<std::size_t>(p), {});
        is_categorical_.assign(static_cast<std::size_t>(p), 0);
        if (categorical_features_.has_value())
            for (int f : *categorical_features_)
                if (f >= 0 && f < static_cast<int>(p))
                    is_categorical_[static_cast<std::size_t>(f)] = 1;

        for (Eigen::Index j = 0; j < p; ++j) {
            std::vector<Scalar> thresholds;
            if (is_categorical_[static_cast<std::size_t>(j)]) {
                // Identity binning: find the max integer category present.
                Scalar max_cat{0};
                for (Eigen::Index i = 0; i < n; ++i)
                    max_cat = std::max(max_cat, X(i, j));
                const int k = std::min(static_cast<int>(max_cat) + 1,
                                       max_bins_);
                for (int b = 1; b < k; ++b)
                    thresholds.push_back(static_cast<Scalar>(b) - Scalar{0.5});
                bin_edges_[static_cast<std::size_t>(j)] = thresholds;
                continue;
            }
            std::vector<Scalar> sorted_col(static_cast<std::size_t>(n));
            for (Eigen::Index i = 0; i < n; ++i) sorted_col[i] = X(i, j);
            std::sort(sorted_col.begin(), sorted_col.end());
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

    // Map X to bin indices using stored thresholds.
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

    // Build per-feature thresholds from a sparse matrix and return the binned
    // uint8 matrix, without materialising a dense double matrix. The implicit
    // zeros are included in the quantile estimate and binned as value 0.
    template <int Options, typename StorageIndex>
    BinnedMatrix build_bins_sparse(
        const Eigen::SparseMatrix<Scalar, Options, StorageIndex>& X) {
        const Eigen::Index n = X.rows();
        const Eigen::Index p = X.cols();
        using ColSparse =
            Eigen::SparseMatrix<Scalar, Eigen::ColMajor, StorageIndex>;
        const ColSparse Xc = X;

        bin_edges_.assign(static_cast<std::size_t>(p), {});
        is_categorical_.assign(static_cast<std::size_t>(p), 0);
        if (categorical_features_.has_value())
            for (int f : *categorical_features_)
                if (f >= 0 && f < static_cast<int>(p))
                    is_categorical_[static_cast<std::size_t>(f)] = 1;

        for (Eigen::Index j = 0; j < p; ++j) {
            std::vector<Scalar> thresholds;
            if (is_categorical_[static_cast<std::size_t>(j)]) {
                Scalar max_cat{0};
                for (typename ColSparse::InnerIterator it(Xc, j); it; ++it)
                    max_cat = std::max(max_cat, it.value());
                const int k = std::min(static_cast<int>(max_cat) + 1,
                                       max_bins_);
                for (int b = 1; b < k; ++b)
                    thresholds.push_back(static_cast<Scalar>(b) - Scalar{0.5});
                bin_edges_[static_cast<std::size_t>(j)] = thresholds;
                continue;
            }
            // Reconstruct the column's full value multiset: explicit nonzeros
            // plus (n - nnz) implicit zeros.
            std::vector<Scalar> col;
            col.reserve(static_cast<std::size_t>(n));
            Eigen::Index nnz = 0;
            for (typename ColSparse::InnerIterator it(Xc, j); it; ++it) {
                col.push_back(it.value());
                ++nnz;
            }
            for (Eigen::Index z = 0; z < n - nnz; ++z)
                col.push_back(Scalar{0});
            std::sort(col.begin(), col.end());
            const int n_thresh = max_bins_ - 1;
            for (int b = 1; b <= n_thresh; ++b) {
                const std::size_t idx = std::min(
                    static_cast<std::size_t>(
                        static_cast<double>(b) * static_cast<double>(n) /
                        static_cast<double>(max_bins_)),
                    col.size() - 1);
                const Scalar t = col[idx];
                if (thresholds.empty() || t > thresholds.back())
                    thresholds.push_back(t);
            }
            bin_edges_[static_cast<std::size_t>(j)] = thresholds;
        }
        return bin_sparse(Xc);
    }

    // Map a sparse matrix to bin indices using stored thresholds. Implicit
    // zeros map to the bin containing value 0.
    template <int Options, typename StorageIndex>
    [[nodiscard]] BinnedMatrix bin_sparse(
        const Eigen::SparseMatrix<Scalar, Options, StorageIndex>& X) const {
        const Eigen::Index n = X.rows();
        const Eigen::Index p = X.cols();
        using ColSparse =
            Eigen::SparseMatrix<Scalar, Eigen::ColMajor, StorageIndex>;
        const ColSparse Xc = X;
        BinnedMatrix Xb(n, p);
        for (Eigen::Index j = 0; j < p; ++j) {
            const auto& thr = bin_edges_[static_cast<std::size_t>(j)];
            // Default every row to the zero-value bin, then overwrite the
            // explicit nonzeros.
            auto z = std::upper_bound(thr.begin(), thr.end(), Scalar{0});
            const uint8_t zero_bin =
                static_cast<uint8_t>(std::distance(thr.begin(), z));
            for (Eigen::Index i = 0; i < n; ++i) Xb(i, j) = zero_bin;
            for (typename ColSparse::InnerIterator it(Xc, j); it; ++it) {
                auto bit = std::upper_bound(thr.begin(), thr.end(), it.value());
                Xb(it.row(), j) =
                    static_cast<uint8_t>(std::distance(thr.begin(), bit));
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
        params.categorical_features = is_categorical_;
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
    std::optional<std::vector<int>> categorical_features_;
    std::optional<std::vector<int>> monotonic_cst_;
    bool early_stopping_;
    Scalar validation_fraction_;
    int n_iter_no_change_;
    Scalar tol_;
    std::optional<uint64_t> random_state_;

    Scalar init_{0};
    std::vector<std::vector<Scalar>> bin_edges_;        // per-feature
    std::vector<int> is_categorical_;                   // per-feature flag
    std::vector<internal::HistTree<Scalar>> estimators_;
    VectorType train_score_;
};

/// @}

} // namespace Skigen

#endif // SKIGEN_ENSEMBLE_HIST_GRADIENT_BOOSTING_REGRESSOR_H
