// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_MODEL_SELECTION_CROSS_VALIDATION_H
#define SKIGEN_MODEL_SELECTION_CROSS_VALIDATION_H

#include <Eigen/Core>
#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace Skigen {

/// cross_val_score — K-fold cross-validation scores.
/// Mirrors sklearn.model_selection.cross_val_score.
///
/// @param estimator Must have fit(X, y) and score(X, y).
/// @param X Feature matrix.
/// @param y Target vector.
/// @param cv Number of folds.
/// @param shuffle Whether to shuffle before splitting.
/// @param random_state Seed for shuffling.
/// @return Vector of scores, one per fold.
template <typename Estimator, typename Scalar>
[[nodiscard]] Eigen::Matrix<Scalar, Eigen::Dynamic, 1>
cross_val_score(
    Estimator estimator,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& X,
    const auto& y,
    int cv = 5,
    bool shuffle = true,
    unsigned int random_state = 42) {

    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    const Eigen::Index n = X.rows();
    if (n < cv) {
        throw std::invalid_argument("n_samples must be >= cv folds.");
    }

    std::vector<Eigen::Index> indices(static_cast<std::size_t>(n));
    std::iota(indices.begin(), indices.end(), Eigen::Index{0});

    if (shuffle) {
        std::mt19937 rng(random_state);
        std::shuffle(indices.begin(), indices.end(), rng);
    }

    VectorType scores(cv);

    for (int fold = 0; fold < cv; ++fold) {
        // Determine fold boundaries
        Eigen::Index fold_start = (n * fold) / cv;
        Eigen::Index fold_end = (n * (fold + 1)) / cv;
        Eigen::Index test_size = fold_end - fold_start;
        Eigen::Index train_size = n - test_size;

        MatrixType X_train(train_size, X.cols());
        MatrixType X_test(test_size, X.cols());

        using YType = std::decay_t<decltype(y)>;
        YType y_train(train_size);
        YType y_test(test_size);

        Eigen::Index train_idx = 0;
        Eigen::Index test_idx = 0;

        for (Eigen::Index i = 0; i < n; ++i) {
            auto src = indices[static_cast<std::size_t>(i)];
            if (i >= fold_start && i < fold_end) {
                X_test.row(test_idx) = X.row(src);
                y_test(test_idx) = y(src);
                ++test_idx;
            } else {
                X_train.row(train_idx) = X.row(src);
                y_train(train_idx) = y(src);
                ++train_idx;
            }
        }

        Estimator est = estimator; // copy
        est.fit(X_train, y_train);
        scores(fold) = static_cast<Scalar>(est.score(X_test, y_test));
    }

    return scores;
}

} // namespace Skigen

#endif // SKIGEN_MODEL_SELECTION_CROSS_VALIDATION_H
