// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_MANIFOLD_MDS_H
#define SKIGEN_MANIFOLD_MDS_H

#include "../Core/Base.h"
#include "../Core/Validation.h"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <random>

namespace Skigen {

/// @defgroup Algo_MDS MDS
/// @ingroup Manifold
/// @brief Multidimensional Scaling (MDS) via SMACOF.
/// @{

/// @brief Multidimensional Scaling (MDS).
///
/// Non-linear dimensionality reduction through the SMACOF (Scaling by
/// MAjorizing a COmplicated Function) algorithm.  Given an input data
/// matrix X the algorithm minimises the *stress* criterion, which
/// measures how well pairwise Euclidean distances in the low-dimensional
/// embedding approximate the original pairwise distances.
///
/// Mirrors
/// [sklearn.manifold.MDS](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html).
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `n_components` | `int` | `2` | Dimensionality of the embedding. |
/// | `max_iter` | `int` | `300` | Maximum number of SMACOF iterations. |
/// | `tol` | `Scalar` | `1e-3` | Convergence tolerance on stress change. |
/// | `metric` | `bool` | `true` | Only metric MDS is supported. |
/// | `random_state` | `std::optional<uint64_t>` | `nullopt` | Seed for the RNG used to initialise positions. |
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type | Description |
/// |----------|------|-------------|
/// | `embedding()` | `MatrixType` | Embedding coordinates (n_samples x n_components). |
/// | `stress()` | `Scalar` | Final raw stress value. |
/// | `n_iter()` | `int` | Number of iterations actually performed. |
///
/// ### Limitations relative to scikit-learn
///
/// Only metric MDS is implemented (`metric=true`).  The parameters
/// `dissimilarity`, `n_init`, `n_jobs`, `normalized_stress`, and
/// `verbose` are not supported.
///
/// ### Examples
///
/// @snippet mds.cpp example_mds
template <typename Scalar = double>
class MDS
    : public Transformer<MDS<Scalar>, Scalar> {
public:
    using Base = Transformer<MDS<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;

    /// @brief Construct an MDS estimator.
    ///
    /// @param n_components Embedding dimensionality (default 2).
    /// @param max_iter Maximum SMACOF iterations (default 300).
    /// @param tol Convergence tolerance on relative stress change (default 1e-3).
    /// @param metric If true, metric MDS is used (default true, the only mode supported).
    /// @param random_state Optional seed for reproducible initialisation.
    explicit MDS(int n_components = 2,
                 int max_iter = 300,
                 Scalar tol = Scalar{1e-3},
                 bool metric = true,
                 std::optional<uint64_t> random_state = std::nullopt)
        : n_components_(n_components),
          max_iter_(max_iter),
          tol_(tol),
          metric_(metric),
          random_state_(random_state) {}

    // -- Accessors ----------------------------------------------------------

    /// @brief Embedding coordinates (n_samples x n_components).
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] const MatrixType& embedding() const {
        this->check_is_fitted(); return embedding_;
    }
    /// @brief Final stress value after fitting.
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] Scalar stress() const {
        this->check_is_fitted(); return stress_;
    }
    /// @brief Number of SMACOF iterations performed.
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] int n_iter() const {
        this->check_is_fitted(); return n_iter_;
    }

    SKIGEN_PARAMS(
        (n_components, n_components_, int),
        (max_iter, max_iter_, int),
        (tol, tol_, double),
        (metric, metric_, bool))

    // -- Implementation (called by CRTP base) --------------------------------

    /// @brief Fit the MDS model using the SMACOF algorithm.
    ///
    /// Computes pairwise Euclidean distances of X then iteratively
    /// optimises an n_components-dimensional embedding that preserves
    /// those distances as faithfully as possible.
    ///
    /// @param X Data matrix of shape (n_samples, n_features).
    /// @return Reference to the fitted transformer (`*this`).
    MDS& fit_impl(const Eigen::Ref<const MatrixType>& X) {
        internal::check_non_empty(X);

        this->n_features_in_ = X.cols();
        const Eigen::Index n = X.rows();
        constexpr Scalar eps = std::numeric_limits<Scalar>::epsilon();

        // 1. Pairwise Euclidean distance matrix D (n x n).
        MatrixType D = pairwise_distances(X);

        // Denominator for stress: sum(D^2).
        const Scalar sum_D_sq = D.array().square().sum();

        // 2. Initialise embedding Y randomly.
        std::mt19937_64 rng(random_state_.value_or(
            std::random_device{}()));
        std::normal_distribution<Scalar> dist(Scalar{0}, Scalar{1});

        MatrixType Y(n, n_components_);
        for (Eigen::Index i = 0; i < n; ++i)
            for (Eigen::Index j = 0; j < n_components_; ++j)
                Y(i, j) = dist(rng);

        // 3. SMACOF iterations.
        Scalar prev_stress = std::numeric_limits<Scalar>::max();
        n_iter_ = 0;

        for (int iter = 0; iter < max_iter_; ++iter) {
            // a. Distance matrix of current positions.
            MatrixType D_Y = pairwise_distances(Y);

            // b. Guttman transform matrix B.
            MatrixType B = MatrixType::Zero(n, n);
            for (Eigen::Index i = 0; i < n; ++i) {
                for (Eigen::Index j = 0; j < n; ++j) {
                    if (i == j) continue;
                    Scalar d_y = std::max(D_Y(i, j), eps);
                    B(i, j) = -D(i, j) / d_y;
                }
                // Diagonal: negative row-sum of off-diagonal entries.
                B(i, i) = -B.row(i).sum() + B(i, i);
            }

            // c. Update positions.
            Y = (Scalar{1} / static_cast<Scalar>(n)) * B * Y;

            // d. Stress = sqrt( sum((D - D_Y)^2) / sum(D^2) ).
            D_Y = pairwise_distances(Y);
            stress_ = (sum_D_sq > Scalar{0})
                ? std::sqrt((D - D_Y).array().square().sum() / sum_D_sq)
                : Scalar{0};
            n_iter_ = iter + 1;

            // e. Convergence check.
            if (std::abs(prev_stress - stress_) < tol_) break;
            prev_stress = stress_;
        }

        embedding_ = std::move(Y);
        this->fitted_ = true;
        return *this;
    }

    /// @brief Return the stored embedding.
    ///
    /// For MDS the transform simply returns the embedding computed
    /// during fit.  The input X is expected to match the training data.
    ///
    /// @param X Data matrix (ignored; the stored embedding is returned).
    /// @return The embedding matrix (n_samples x n_components).
    [[nodiscard]] MatrixType transform_impl(
        const Eigen::Ref<const MatrixType>& /*X*/) const {
        return embedding_;
    }

private:
    int n_components_;
    int max_iter_;
    Scalar tol_;
    bool metric_;
    std::optional<uint64_t> random_state_;

    MatrixType embedding_;
    Scalar stress_ = Scalar{0};
    int n_iter_ = 0;

    /// @brief Compute the pairwise Euclidean distance matrix.
    ///
    /// For an (n x p) matrix, returns an (n x n) symmetric matrix where
    /// entry (i, j) is ||row_i - row_j||_2.
    static MatrixType pairwise_distances(const Eigen::Ref<const MatrixType>& M) {
        const Eigen::Index n = M.rows();
        // Squared norms of each row.
        VectorType sq = M.rowwise().squaredNorm();
        // D^2(i,j) = ||x_i||^2 + ||x_j||^2 - 2 x_i . x_j
        MatrixType D2 = sq.replicate(1, n) + sq.transpose().replicate(n, 1)
                        - Scalar{2} * (M * M.transpose());
        // Clamp negative values arising from floating-point error.
        return D2.array().max(Scalar{0}).sqrt().matrix();
    }
};

/// @}

} // namespace Skigen

#endif // SKIGEN_MANIFOLD_MDS_H
