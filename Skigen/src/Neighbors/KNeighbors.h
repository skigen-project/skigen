// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_NEIGHBORS_KNEIGHBORS_CLASSIFIER_H
#define SKIGEN_NEIGHBORS_KNEIGHBORS_CLASSIFIER_H

#include "../Core/Validation.h"

#include <Eigen/Core>
#include <algorithm>
#include <map>
#include <stdexcept>
#include <vector>

namespace Skigen {

/// KNeighborsClassifier — brute-force k-nearest neighbors classifier.
/// Mirrors sklearn.neighbors.KNeighborsClassifier.
template <typename Scalar = double>
class KNeighborsClassifier {
public:
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using IndexVector = Eigen::VectorXi;

    explicit KNeighborsClassifier(int n_neighbors = 5)
        : n_neighbors_(n_neighbors) {}

    // -- Accessors ----------------------------------------------------------

    [[nodiscard]] bool is_fitted() const noexcept { return fitted_; }
    [[nodiscard]] int n_neighbors() const noexcept { return n_neighbors_; }

    // -- fit / predict -------------------------------------------------------

    KNeighborsClassifier& fit(const Eigen::Ref<const MatrixType>& X,
                              const Eigen::Ref<const IndexVector>& y) {
        internal::check_non_empty(X);
        if (X.rows() != y.rows()) {
            throw std::invalid_argument("X and y have inconsistent lengths.");
        }
        if (X.rows() < n_neighbors_) {
            throw std::invalid_argument(
                "n_samples must be >= n_neighbors.");
        }

        X_train_ = X;
        y_train_ = y;
        n_features_in_ = X.cols();
        fitted_ = true;
        return *this;
    }

    [[nodiscard]] IndexVector predict(
        const Eigen::Ref<const MatrixType>& X) const {
        if (!fitted_) throw std::runtime_error(
            "KNeighborsClassifier has not been fitted yet.");
        if (X.cols() != n_features_in_) {
            throw std::invalid_argument("Feature count mismatch.");
        }

        IndexVector predictions(X.rows());

        // For each query point, find k nearest neighbors and vote
        std::vector<std::pair<Scalar, int>> dists(
            static_cast<std::size_t>(X_train_.rows()));

        for (Eigen::Index i = 0; i < X.rows(); ++i) {
            for (Eigen::Index j = 0; j < X_train_.rows(); ++j) {
                dists[static_cast<std::size_t>(j)] = {
                    (X.row(i) - X_train_.row(j)).squaredNorm(),
                    y_train_(j)
                };
            }

            std::partial_sort(dists.begin(),
                              dists.begin() + n_neighbors_,
                              dists.end());

            // Majority vote
            std::map<int, int> votes;
            for (int k = 0; k < n_neighbors_; ++k) {
                votes[dists[static_cast<std::size_t>(k)].second]++;
            }

            int best_label = 0;
            int best_count = 0;
            for (const auto& [label, count] : votes) {
                if (count > best_count) {
                    best_count = count;
                    best_label = label;
                }
            }
            predictions(i) = best_label;
        }

        return predictions;
    }

    [[nodiscard]] Scalar score(const Eigen::Ref<const MatrixType>& X,
                               const Eigen::Ref<const IndexVector>& y) const {
        IndexVector preds = predict(X);
        int correct = 0;
        for (Eigen::Index i = 0; i < y.size(); ++i) {
            if (preds(i) == y(i)) ++correct;
        }
        return static_cast<Scalar>(correct) / static_cast<Scalar>(y.size());
    }

private:
    int n_neighbors_;
    bool fitted_ = false;
    Eigen::Index n_features_in_ = 0;

    MatrixType X_train_;
    IndexVector y_train_;
};

/// KNeighborsRegressor — brute-force k-nearest neighbors regressor.
/// Mirrors sklearn.neighbors.KNeighborsRegressor.
template <typename Scalar = double>
class KNeighborsRegressor {
public:
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    explicit KNeighborsRegressor(int n_neighbors = 5)
        : n_neighbors_(n_neighbors) {}

    [[nodiscard]] bool is_fitted() const noexcept { return fitted_; }

    KNeighborsRegressor& fit(const Eigen::Ref<const MatrixType>& X,
                             const Eigen::Ref<const VectorType>& y) {
        internal::check_non_empty(X);
        if (X.rows() != y.rows()) {
            throw std::invalid_argument("X and y have inconsistent lengths.");
        }

        X_train_ = X;
        y_train_ = y;
        n_features_in_ = X.cols();
        fitted_ = true;
        return *this;
    }

    [[nodiscard]] VectorType predict(
        const Eigen::Ref<const MatrixType>& X) const {
        if (!fitted_) throw std::runtime_error(
            "KNeighborsRegressor has not been fitted yet.");

        VectorType predictions(X.rows());
        std::vector<std::pair<Scalar, Scalar>> dists(
            static_cast<std::size_t>(X_train_.rows()));

        for (Eigen::Index i = 0; i < X.rows(); ++i) {
            for (Eigen::Index j = 0; j < X_train_.rows(); ++j) {
                dists[static_cast<std::size_t>(j)] = {
                    (X.row(i) - X_train_.row(j)).squaredNorm(),
                    y_train_(j)
                };
            }

            std::partial_sort(dists.begin(),
                              dists.begin() + n_neighbors_,
                              dists.end());

            Scalar sum{0};
            for (int k = 0; k < n_neighbors_; ++k) {
                sum += dists[static_cast<std::size_t>(k)].second;
            }
            predictions(i) = sum / static_cast<Scalar>(n_neighbors_);
        }

        return predictions;
    }

    [[nodiscard]] Scalar score(const Eigen::Ref<const MatrixType>& X,
                               const Eigen::Ref<const VectorType>& y) const {
        VectorType y_pred = predict(X);
        Scalar ss_res = (y - y_pred).squaredNorm();
        Scalar ss_tot = (y.array() - y.mean()).matrix().squaredNorm();
        if (ss_tot == Scalar{0}) return Scalar{0};
        return Scalar{1} - ss_res / ss_tot;
    }

private:
    int n_neighbors_;
    bool fitted_ = false;
    Eigen::Index n_features_in_ = 0;

    MatrixType X_train_;
    VectorType y_train_;
};

} // namespace Skigen

#endif // SKIGEN_NEIGHBORS_KNEIGHBORS_CLASSIFIER_H
