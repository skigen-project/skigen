// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

#ifndef SKIGEN_TREE_DETAIL_FEATURE_COLUMNS_H
#define SKIGEN_TREE_DETAIL_FEATURE_COLUMNS_H

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <vector>

namespace Skigen {
namespace internal {

/// @brief Dense column accessor for the decision-tree split finder.
///
/// Thin wrapper over a dense design matrix. `operator()(row, f)` returns the
/// cell value directly, so the split-finding loop sees exactly the same
/// values it would with raw `X(row, f)` access — guaranteeing bit-for-bit
/// identical trees on dense input.
template <typename Scalar>
class DenseFeatureColumns {
public:
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    explicit DenseFeatureColumns(const Eigen::Ref<const Matrix>& X) : X_(X) {}

    [[nodiscard]] Scalar operator()(Eigen::Index row, Eigen::Index f) const {
        return X_(row, f);
    }
    [[nodiscard]] Eigen::Index rows() const noexcept { return X_.rows(); }
    [[nodiscard]] Eigen::Index cols() const noexcept { return X_.cols(); }

private:
    Eigen::Ref<const Matrix> X_;
};

/// @brief Native sparse column accessor for the decision-tree split finder.
///
/// Holds a column-major (CSC) copy of the sparse design matrix and
/// materialises **one feature column at a time** into a dense length-`rows`
/// scratch vector on demand. Implicit zeros are filled in as `0`, so the
/// split finder treats them exactly as scikit-learn's sparse splitter does
/// (zeros participate in the sorted order as value 0). The full dense
/// @f$n \times p@f$ matrix is never materialised — at most one column of
/// length `rows` is held at any time.
template <typename Scalar, int Options, typename StorageIndex>
class SparseFeatureColumns {
public:
    using SparseType = Eigen::SparseMatrix<Scalar, Options, StorageIndex>;
    using ColSparse = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, StorageIndex>;

    explicit SparseFeatureColumns(const SparseType& X)
        : Xc_(X),
          rows_(X.rows()),
          cols_(X.cols()),
          cached_feature_(-1),
          column_(static_cast<std::size_t>(X.rows()), Scalar{0}) {}

    [[nodiscard]] Scalar operator()(Eigen::Index row, Eigen::Index f) const {
        if (f != cached_feature_) load_column(f);
        return column_[static_cast<std::size_t>(row)];
    }
    [[nodiscard]] Eigen::Index rows() const noexcept { return rows_; }
    [[nodiscard]] Eigen::Index cols() const noexcept { return cols_; }

private:
    void load_column(Eigen::Index f) const {
        std::fill(column_.begin(), column_.end(), Scalar{0});
        for (typename ColSparse::InnerIterator it(Xc_, f); it; ++it)
            column_[static_cast<std::size_t>(it.row())] = it.value();
        cached_feature_ = f;
    }

    ColSparse Xc_;
    Eigen::Index rows_;
    Eigen::Index cols_;
    mutable Eigen::Index cached_feature_;
    mutable std::vector<Scalar> column_;
};

}  // namespace internal
}  // namespace Skigen

#endif  // SKIGEN_TREE_DETAIL_FEATURE_COLUMNS_H
