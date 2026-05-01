// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_MODEL_SELECTION_TRAIN_TEST_SPLIT_H
#define SKIGEN_MODEL_SELECTION_TRAIN_TEST_SPLIT_H

#include <Eigen/Core>
#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>
#include <tuple>

namespace Skigen {

/// Result of train_test_split for (X, y) data.
template <typename MatrixType, typename LabelType>
struct TrainTestSplitResult {
    MatrixType X_train;
    MatrixType X_test;
    LabelType y_train;
    LabelType y_test;
};

/// train_test_split — Split arrays into random train and test subsets.
/// Mirrors sklearn.model_selection.train_test_split.
///
/// @param X        Feature matrix (n_samples x n_features).
/// @param y        Label vector (n_samples).
/// @param test_size Fraction of data for test set (0, 1).
/// @param random_state Seed for shuffling.
/// @param shuffle  Whether to shuffle before splitting.
template <typename Scalar, typename LabelType>
[[nodiscard]] auto train_test_split(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& X,
    const LabelType& y,
    double test_size = 0.25,
    unsigned int random_state = 42,
    bool shuffle = true) {

    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    if (X.rows() != y.rows()) {
        throw std::invalid_argument("X and y must have the same number of rows.");
    }
    if (test_size <= 0.0 || test_size >= 1.0) {
        throw std::invalid_argument("test_size must be in (0, 1).");
    }

    const Eigen::Index n = X.rows();
    const Eigen::Index n_test = static_cast<Eigen::Index>(
        std::round(static_cast<double>(n) * test_size));
    const Eigen::Index n_train = n - n_test;

    if (n_train <= 0 || n_test <= 0) {
        throw std::invalid_argument("Split produces empty train or test set.");
    }

    std::vector<Eigen::Index> indices(static_cast<std::size_t>(n));
    std::iota(indices.begin(), indices.end(), Eigen::Index{0});

    if (shuffle) {
        std::mt19937 rng(random_state);
        std::shuffle(indices.begin(), indices.end(), rng);
    }

    MatrixType X_train(n_train, X.cols());
    MatrixType X_test(n_test, X.cols());
    LabelType y_train(n_train);
    LabelType y_test(n_test);

    for (Eigen::Index i = 0; i < n_train; ++i) {
        X_train.row(i) = X.row(indices[static_cast<std::size_t>(i)]);
        y_train(i) = y(indices[static_cast<std::size_t>(i)]);
    }
    for (Eigen::Index i = 0; i < n_test; ++i) {
        auto idx = indices[static_cast<std::size_t>(n_train + i)];
        X_test.row(i) = X.row(idx);
        y_test(i) = y(idx);
    }

    return TrainTestSplitResult<MatrixType, LabelType>{
        std::move(X_train), std::move(X_test),
        std::move(y_train), std::move(y_test)};
}

} // namespace Skigen

#endif // SKIGEN_MODEL_SELECTION_TRAIN_TEST_SPLIT_H
