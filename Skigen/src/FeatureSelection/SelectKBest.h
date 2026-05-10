// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_FEATURE_SELECTION_SELECT_K_BEST_H
#define SKIGEN_FEATURE_SELECTION_SELECT_K_BEST_H

#include "../Core/Base.h"
#include "../Core/Validation.h"
#include "ScoreFunctions.h"

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace Skigen {

namespace feature_selection {

// Default callable for f_classif so users can construct without specifying
// a score function explicitly.
template <typename Scalar>
struct FClassif {
    using RowVec = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;
    using Mat    = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    std::pair<RowVec, RowVec> operator()(
        const Eigen::Ref<const Mat>& X,
        const Eigen::Ref<const Eigen::VectorXi>& y) const {
        return f_classif<Scalar>(X, y);
    }
};

template <typename Scalar>
struct FRegression {
    using RowVec = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;
    using Mat    = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vec    = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    std::pair<RowVec, RowVec> operator()(
        const Eigen::Ref<const Mat>& X,
        const Eigen::Ref<const Vec>& y) const {
        return f_regression<Scalar>(X, y);
    }
};

template <typename Scalar>
struct Chi2 {
    using RowVec = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;
    using Mat    = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    std::pair<RowVec, RowVec> operator()(
        const Eigen::Ref<const Mat>& X,
        const Eigen::Ref<const Eigen::VectorXi>& y) const {
        return chi2<Scalar>(X, y);
    }

    // Sparse overload: forwards to the sparse chi2 implementation.
    template <int Options, typename StorageIndex>
    std::pair<RowVec, RowVec> operator()(
        const Eigen::SparseMatrix<Scalar, Options, StorageIndex>& X,
        const Eigen::Ref<const Eigen::VectorXi>& y) const {
        return chi2<Scalar, Options, StorageIndex>(X, y);
    }
};

}  // namespace feature_selection

/// @defgroup Algo_SelectKBest SelectKBest
/// @ingroup FeatureSelection
/// @brief Select features according to the k highest scores.
/// @{

/// @brief Select features according to the k highest scores.
///
/// Mirrors
/// [sklearn.feature_selection.SelectKBest](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html).
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `score_func` | `ScoreFn` | `f_classif` | Callable returning `(scores, p-values)` per feature. |
/// | `k` | `int` | `10` | Number of top features to select. Use `-1` to select all features (matching sklearn `"all"`). |
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type | Description |
/// |----------|------|-------------|
/// | `scores()` | `RowVectorType` | Per-feature scores returned by `score_func`. |
/// | `pvalues()` | `RowVectorType` | Per-feature p-values (may be empty if not provided). |
/// | `n_features_in()` | `IndexType` | Number of features seen during fit. |
///
/// ### See also
///
/// - Skigen::feature_selection::f_classif — ANOVA F-test for classification.
/// - Skigen::feature_selection::f_regression — F-test for regression.
/// - Skigen::feature_selection::chi2 — Chi-squared test.
///
/// @note **scikit-learn parity gaps:** Passing the score function as a string
///   (e.g. `"f_classif"`) is not supported — pass the callable directly.
///   `feature_names_in_` is not exposed.
template <typename Scalar = double,
          typename ScoreFn = feature_selection::FClassif<Scalar>>
class SelectKBest
    : public Estimator<SelectKBest<Scalar, ScoreFn>, Scalar> {
public:
    using Base = Estimator<SelectKBest<Scalar, ScoreFn>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;
    using typename Base::ScalarType;
    using BoolMaskType = Eigen::Array<bool, Eigen::Dynamic, 1>;

    /// @brief Construct a SelectKBest selector.
    explicit SelectKBest(ScoreFn score_func = ScoreFn{}, int k = 10)
        : score_func_(std::move(score_func)), k_(k) {}

    // -- Accessors ----------------------------------------------------------

    [[nodiscard]] int k() const noexcept { return k_; }

    [[nodiscard]] const RowVectorType& scores() const {
        this->check_is_fitted();
        return scores_;
    }

    [[nodiscard]] const RowVectorType& pvalues() const {
        this->check_is_fitted();
        return pvalues_;
    }

    [[nodiscard]] BoolMaskType get_support_mask() const {
        this->check_is_fitted();
        return support_mask_;
    }

    [[nodiscard]] BoolMaskType get_support(bool /*indices*/ = false) const {
        this->check_is_fitted();
        return support_mask_;
    }

    [[nodiscard]] Eigen::VectorXi get_support_indices() const {
        this->check_is_fitted();
        std::vector<int> idx;
        for (Eigen::Index j = 0; j < support_mask_.size(); ++j) {
            if (support_mask_(j)) idx.push_back(static_cast<int>(j));
        }
        Eigen::VectorXi out(idx.size());
        for (std::size_t i = 0; i < idx.size(); ++i) {
            out(static_cast<Eigen::Index>(i)) = idx[i];
        }
        return out;
    }

    // -- Fit overloads ------------------------------------------------------

    /// @brief Fit using a classification target (integer labels).
    SelectKBest& fit(const Eigen::Ref<const MatrixType>& X,
                     const Eigen::Ref<const Eigen::VectorXi>& y) {
        internal::check_non_empty(X);
        internal::check_consistent_length(X, y);
        this->n_features_in_ = X.cols();
        auto [s, p] = score_func_(X, y);
        scores_ = s;
        pvalues_ = p;
        compute_support();
        this->fitted_ = true;
        return *this;
    }

    /// @brief Fit using a regression target (continuous values).
    SelectKBest& fit(const Eigen::Ref<const MatrixType>& X,
                     const Eigen::Ref<const VectorType>& y) {
        internal::check_non_empty(X);
        internal::check_consistent_length(X, y);
        this->n_features_in_ = X.cols();
        auto [s, p] = score_func_(X, y);
        scores_ = s;
        pvalues_ = p;
        compute_support();
        this->fitted_ = true;
        return *this;
    }

    // -- Sparse-aware overloads (v1.1.0 §3.2) --------------------------------

    /// @brief Fit using a sparse design matrix and a classification target.
    ///
    /// Only compiles when the configured `ScoreFn` provides a sparse
    /// `operator()` overload — currently `Chi2` (and any user-supplied
    /// score function with the same shape). For `FClassif` and
    /// `FRegression`, the sparse path is a documented v1.1.0 parity gap
    /// and a compile error.
    template <int Options, typename StorageIndex>
    SelectKBest& fit(
        const Eigen::SparseMatrix<Scalar, Options, StorageIndex>& X,
        const Eigen::Ref<const Eigen::VectorXi>& y) {
        if (X.rows() == 0 || X.cols() == 0) {
            throw std::invalid_argument(
                "SelectKBest.fit: empty sparse matrix.");
        }
        if (X.rows() != y.size()) {
            throw std::invalid_argument(
                "SelectKBest.fit: X has " + std::to_string(X.rows()) +
                " rows but y has " + std::to_string(y.size()) + " entries.");
        }
        this->n_features_in_ = X.cols();
        auto [s, p] = score_func_(X, y);
        scores_ = s;
        pvalues_ = p;
        compute_support();
        this->fitted_ = true;
        return *this;
    }

    /// @brief Transform a sparse design matrix to keep only the top-k columns.
    template <int Options, typename StorageIndex>
    [[nodiscard]] Eigen::SparseMatrix<Scalar, Options, StorageIndex>
    transform(
        const Eigen::SparseMatrix<Scalar, Options, StorageIndex>& X) const {
        this->check_is_fitted();
        if (X.cols() != this->n_features_in_) {
            throw std::invalid_argument(
                "X has " + std::to_string(X.cols()) +
                " features, but selector was fitted with " +
                std::to_string(this->n_features_in_) + " features.");
        }

        std::vector<Eigen::Index> kept;
        kept.reserve(static_cast<std::size_t>(support_mask_.size()));
        for (Eigen::Index j = 0; j < support_mask_.size(); ++j) {
            if (support_mask_(j)) kept.push_back(j);
        }

        using ColSparse =
            Eigen::SparseMatrix<Scalar, Eigen::ColMajor, StorageIndex>;
        const ColSparse Xc = X;
        ColSparse out_col(X.rows(), static_cast<Eigen::Index>(kept.size()));
        std::vector<Eigen::Index> col_nnz(kept.size(), 0);
        for (std::size_t i = 0; i < kept.size(); ++i) {
            col_nnz[i] =
                Xc.outerIndexPtr()[kept[i] + 1] - Xc.outerIndexPtr()[kept[i]];
        }
        out_col.reserve(col_nnz);
        for (std::size_t i = 0; i < kept.size(); ++i) {
            for (typename ColSparse::InnerIterator it(Xc, kept[i]); it; ++it) {
                out_col.insert(it.row(), static_cast<Eigen::Index>(i)) =
                    it.value();
            }
        }
        out_col.makeCompressed();
        return Eigen::SparseMatrix<Scalar, Options, StorageIndex>(out_col);
    }

    // -- Transform ----------------------------------------------------------

    [[nodiscard]] MatrixType transform(
        const Eigen::Ref<const MatrixType>& X) const {
        this->check_is_fitted();
        this->validate_feature_count(X);
        Eigen::Index n_kept = 0;
        for (Eigen::Index j = 0; j < support_mask_.size(); ++j) {
            if (support_mask_(j)) ++n_kept;
        }
        MatrixType out(X.rows(), n_kept);
        Eigen::Index k = 0;
        for (Eigen::Index j = 0; j < support_mask_.size(); ++j) {
            if (support_mask_(j)) out.col(k++) = X.col(j);
        }
        return out;
    }

    template <typename YType>
    [[nodiscard]] MatrixType fit_transform(
        const Eigen::Ref<const MatrixType>& X, const YType& y) {
        fit(X, y);
        return transform(X);
    }

    [[nodiscard]] MatrixType inverse_transform(
        const Eigen::Ref<const MatrixType>& X) const {
        this->check_is_fitted();
        Eigen::Index n_kept = 0;
        for (Eigen::Index j = 0; j < support_mask_.size(); ++j) {
            if (support_mask_(j)) ++n_kept;
        }
        if (X.cols() != n_kept) {
            throw std::invalid_argument(
                "X has " + std::to_string(X.cols()) +
                " features but selector kept " + std::to_string(n_kept) + ".");
        }
        MatrixType out = MatrixType::Zero(X.rows(), this->n_features_in_);
        Eigen::Index k = 0;
        for (Eigen::Index j = 0; j < this->n_features_in_; ++j) {
            if (support_mask_(j)) out.col(j) = X.col(k++);
        }
        return out;
    }

private:
    void compute_support() {
        const Eigen::Index p = scores_.size();
        support_mask_ = BoolMaskType::Constant(p, false);
        int k_eff = (k_ < 0) ? static_cast<int>(p)
                             : std::min<int>(k_, static_cast<int>(p));
        if (k_eff == 0) return;

        // Sort indices by descending score (NaN treated as -inf).
        std::vector<Eigen::Index> idx(static_cast<std::size_t>(p));
        for (Eigen::Index j = 0; j < p; ++j)
            idx[static_cast<std::size_t>(j)] = j;
        std::sort(idx.begin(), idx.end(),
                  [&](Eigen::Index a, Eigen::Index b) {
                      Scalar sa = scores_(a);
                      Scalar sb = scores_(b);
                      bool nana = !(sa == sa);
                      bool nanb = !(sb == sb);
                      if (nana && !nanb) return false;
                      if (!nana && nanb) return true;
                      if (nana && nanb) return a < b;
                      if (sa == sb) return a < b;
                      return sa > sb;
                  });
        for (int i = 0; i < k_eff; ++i) {
            support_mask_(idx[static_cast<std::size_t>(i)]) = true;
        }
    }

    ScoreFn score_func_;
    int k_;
    RowVectorType scores_;
    RowVectorType pvalues_;
    BoolMaskType support_mask_;
};

/// @}

// Convenience type aliases mirroring TypeAliases pattern.
template <typename Scalar = double>
using SelectKBestFClassif =
    SelectKBest<Scalar, feature_selection::FClassif<Scalar>>;
template <typename Scalar = double>
using SelectKBestFRegression =
    SelectKBest<Scalar, feature_selection::FRegression<Scalar>>;
template <typename Scalar = double>
using SelectKBestChi2 = SelectKBest<Scalar, feature_selection::Chi2<Scalar>>;

}  // namespace Skigen

#endif  // SKIGEN_FEATURE_SELECTION_SELECT_K_BEST_H
