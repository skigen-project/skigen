// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_MANIFOLD_TSNE_H
#define SKIGEN_MANIFOLD_TSNE_H

#include "../Core/Base.h"
#include "../Core/Validation.h"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <optional>
#include <random>

namespace Skigen {

/// @defgroup Algo_Manifold Manifold Learning
/// @ingroup Manifold
/// @brief Manifold learning estimators for nonlinear dimensionality reduction.
/// @{

/// @brief t-Distributed Stochastic Neighbor Embedding (t-SNE).
///
/// Nonlinear dimensionality reduction that embeds high-dimensional data
/// in a low-dimensional space by modelling pairwise similarities as
/// joint probabilities with Gaussian kernels in the input space and a
/// Student-t kernel in the embedding space.
///
/// Mirrors
/// [sklearn.manifold.TSNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html).
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `n_components` | `int` | `2` | Dimension of the embedded space. |
/// | `perplexity` | `Scalar` | `30.0` | Related to the number of nearest neighbors. |
/// | `learning_rate` | `Scalar` | `200.0` | Step size for gradient descent. |
/// | `n_iter` | `int` | `1000` | Maximum number of gradient descent iterations. |
/// | `random_state` | `std::optional<uint64_t>` | `nullopt` | Seed for reproducibility. |
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type | Description |
/// |----------|------|-------------|
/// | `embedding()` | `const MatrixType&` | Embedding vectors (n_samples x n_components). |
/// | `kl_divergence()` | `Scalar` | Final Kullback-Leibler divergence. |
///
/// ### Algorithm — exact method
///
/// 1. Compute pairwise squared Euclidean distances.
/// 2. Binary-search per-point bandwidths to match the target perplexity.
/// 3. Symmetrise conditional probabilities to obtain the joint distribution P.
/// 4. Gradient descent with momentum minimises the KL divergence between
///    P and the Student-t joint distribution Q in the embedding space.
///    Early exaggeration (factor 4) is applied for the first 250 iterations.
///
/// ### Limitations relative to scikit-learn
///
/// Only the exact method is implemented (no Barnes-Hut approximation).
/// The following scikit-learn parameters are not supported: `method`,
/// `metric`, `init`, `early_exaggeration`, `min_grad_norm`, `angle`,
/// `n_jobs`, `verbose`.
///
/// ### Examples
///
/// @snippet tsne.cpp example_tsne
template <typename Scalar = double>
class TSNE
    : public Transformer<TSNE<Scalar>, Scalar> {
public:
    using Base = Transformer<TSNE<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;

    /// @brief Construct a t-SNE estimator.
    ///
    /// @param n_components Dimension of the embedded space (default `2`).
    /// @param perplexity Related to the number of nearest neighbors
    ///   (default `30.0`).
    /// @param learning_rate Gradient descent step size (default `200.0`).
    /// @param n_iter Maximum number of iterations (default `1000`).
    /// @param random_state Optional RNG seed for reproducibility.
    explicit TSNE(int n_components = 2,
                  Scalar perplexity = Scalar{30},
                  Scalar learning_rate = Scalar{200},
                  int n_iter = 1000,
                  std::optional<uint64_t> random_state = std::nullopt)
        : n_components_(n_components)
        , perplexity_(perplexity)
        , learning_rate_(learning_rate)
        , n_iter_(n_iter)
        , random_state_(random_state) {}

    // -- Accessors ----------------------------------------------------------

    /// @brief Embedding vectors of shape (n_samples, n_components).
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] const MatrixType& embedding() const {
        this->check_is_fitted();
        return embedding_;
    }

    /// @brief Final KL divergence between P and Q.
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] Scalar kl_divergence() const {
        this->check_is_fitted();
        return kl_divergence_;
    }

    SKIGEN_PARAMS(
        (n_components, n_components_, int),
        (perplexity,   perplexity_,   double),
        (learning_rate, learning_rate_, double),
        (n_iter,       n_iter_,       int))

    // -- Implementation (called by CRTP base) --------------------------------

    /// @brief Fit the t-SNE model: compute an embedding of X.
    ///
    /// @param X Input data of shape (n_samples, n_features).
    /// @return Reference to the fitted transformer (`*this`).
    TSNE& fit_impl(const Eigen::Ref<const MatrixType>& X) {
        internal::check_non_empty(X);

        this->n_features_in_ = X.cols();
        const Eigen::Index n = X.rows();

        // --- Step 1: Pairwise squared Euclidean distances ---
        MatrixType D = pairwise_squared_distances(X);

        // --- Step 2: Joint probabilities P ---
        MatrixType P = compute_joint_probabilities(D, n);

        // --- Step 3: Initialise Y from N(0, 1e-4) ---
        std::mt19937_64 rng(random_state_.value_or(
            static_cast<uint64_t>(std::random_device{}())));
        std::normal_distribution<Scalar> dist(Scalar{0}, Scalar{1e-4});

        embedding_.resize(n, n_components_);
        for (Eigen::Index i = 0; i < n; ++i) {
            for (Eigen::Index j = 0; j < n_components_; ++j) {
                embedding_(i, j) = dist(rng);
            }
        }

        // --- Step 4: Gradient descent with momentum ---
        MatrixType Y_prev = MatrixType::Zero(n, n_components_);
        MatrixType gains = MatrixType::Ones(n, n_components_);

        constexpr int early_exag_stop = 250;
        constexpr Scalar early_exag_factor = Scalar{4};
        constexpr Scalar momentum_early = Scalar{0.5};
        constexpr Scalar momentum_late = Scalar{0.8};

        // Apply early exaggeration to a working copy of P.
        MatrixType P_work = P * early_exag_factor;

        for (int iter = 0; iter < n_iter_; ++iter) {
            // Turn off early exaggeration after 250 iterations.
            if (iter == early_exag_stop) {
                P_work = P;
            }

            Scalar momentum = (iter < early_exag_stop)
                                   ? momentum_early
                                   : momentum_late;

            // Compute Student-t joint distribution Q.
            MatrixType D_Y = pairwise_squared_distances(embedding_);
            MatrixType num = (MatrixType::Ones(n, n) + D_Y).cwiseInverse();
            // Zero the diagonal.
            num.diagonal().setZero();

            Scalar sum_num = num.sum();
            if (sum_num < Scalar{1e-12}) sum_num = Scalar{1e-12};
            MatrixType Q = num / sum_num;
            // Clamp Q for numerical stability.
            Q = Q.cwiseMax(Scalar{1e-12});

            // Gradient: dY_i = 4 * sum_j (P_ij - Q_ij)(y_i - y_j)(1+||y_i-y_j||^2)^{-1}
            MatrixType PQ = P_work - Q;
            MatrixType dY = MatrixType::Zero(n, n_components_);
            for (Eigen::Index i = 0; i < n; ++i) {
                for (Eigen::Index j = 0; j < n; ++j) {
                    if (i == j) continue;
                    Scalar factor = Scalar{4} * PQ(i, j) * num(i, j);
                    dY.row(i) += factor * (embedding_.row(i) - embedding_.row(j));
                }
            }

            // Adaptive gain (sign-based) to accelerate convergence.
            MatrixType update = embedding_ - Y_prev;
            for (Eigen::Index i = 0; i < n; ++i) {
                for (Eigen::Index j = 0; j < n_components_; ++j) {
                    bool same_sign = (dY(i, j) > Scalar{0}) ==
                                     (update(i, j) > Scalar{0});
                    gains(i, j) = same_sign
                                      ? std::max(gains(i, j) * Scalar{0.8},
                                                 Scalar{0.01})
                                      : gains(i, j) + Scalar{0.2};
                }
            }

            MatrixType Y_old = embedding_;
            embedding_ = embedding_ - learning_rate_ * (gains.cwiseProduct(dY))
                         + momentum * update;

            // Center embedding at the origin.
            RowVectorType mean = embedding_.colwise().mean();
            embedding_.rowwise() -= mean;

            Y_prev = Y_old;
        }

        // Compute final KL divergence.
        kl_divergence_ = compute_kl_divergence(P, embedding_, n);

        this->fitted_ = true;
        return *this;
    }

    /// @brief Return the stored embedding (same as fit result).
    ///
    /// Because t-SNE has no parametric mapping, `transform` simply returns
    /// the embedding computed during `fit`. X is expected to be the same
    /// data that was used for fitting.
    ///
    /// @param X Data matrix (must match the training data shape).
    /// @return The embedding of shape (n_samples, n_components).
    [[nodiscard]] MatrixType transform_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        (void)X;
        return embedding_;
    }

private:
    int n_components_;
    Scalar perplexity_;
    Scalar learning_rate_;
    int n_iter_;
    std::optional<uint64_t> random_state_;

    MatrixType embedding_;
    Scalar kl_divergence_ = Scalar{0};

    // -- Helpers ------------------------------------------------------------

    /// @brief Compute the n x n matrix of pairwise squared Euclidean distances.
    static MatrixType pairwise_squared_distances(
        const Eigen::Ref<const MatrixType>& X) {
        const Eigen::Index n = X.rows();
        // ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i . x_j
        VectorType sq_norms = X.rowwise().squaredNorm();
        MatrixType D = sq_norms.replicate(1, n)
                       + sq_norms.transpose().replicate(n, 1)
                       - Scalar{2} * (X * X.transpose());
        D = D.cwiseMax(Scalar{0});
        return D;
    }

    /// @brief Binary search for per-point Gaussian bandwidth and build P.
    ///
    /// For each point i the bandwidth sigma_i is chosen so that the
    /// Shannon entropy of the conditional distribution p(j|i) matches
    /// log(perplexity).
    MatrixType compute_joint_probabilities(
        const MatrixType& D, Eigen::Index n) const {
        const Scalar target_entropy = std::log(perplexity_);
        constexpr Scalar tol = Scalar{1e-5};
        constexpr int max_tries = 50;

        MatrixType P = MatrixType::Zero(n, n);

        for (Eigen::Index i = 0; i < n; ++i) {
            Scalar beta_min = -std::numeric_limits<Scalar>::infinity();
            Scalar beta_max =  std::numeric_limits<Scalar>::infinity();
            // beta = 1 / (2 sigma^2)
            Scalar beta = Scalar{1};

            for (int attempt = 0; attempt < max_tries; ++attempt) {
                // Compute p(j|i) with current beta.
                VectorType p_i(n);
                Scalar sum_p = Scalar{0};
                for (Eigen::Index j = 0; j < n; ++j) {
                    if (j == i) {
                        p_i(j) = Scalar{0};
                    } else {
                        p_i(j) = std::exp(-D(i, j) * beta);
                        sum_p += p_i(j);
                    }
                }
                if (sum_p < Scalar{1e-30}) sum_p = Scalar{1e-30};
                p_i /= sum_p;

                // Shannon entropy H = -sum p_j log p_j
                Scalar entropy = Scalar{0};
                for (Eigen::Index j = 0; j < n; ++j) {
                    if (j != i && p_i(j) > Scalar{1e-30}) {
                        entropy -= p_i(j) * std::log(p_i(j));
                    }
                }

                Scalar diff = entropy - target_entropy;
                if (std::abs(diff) < tol) {
                    P.row(i) = p_i.transpose();
                    break;
                }

                // Binary search update.
                if (diff > Scalar{0}) {
                    // Entropy too high -> increase beta (narrower kernel).
                    beta_min = beta;
                    beta = std::isinf(beta_max) ? beta * Scalar{2}
                                                : (beta + beta_max) / Scalar{2};
                } else {
                    // Entropy too low -> decrease beta (wider kernel).
                    beta_max = beta;
                    beta = std::isinf(beta_min) ? beta / Scalar{2}
                                                : (beta + beta_min) / Scalar{2};
                }

                if (attempt == max_tries - 1) {
                    P.row(i) = p_i.transpose();
                }
            }
        }

        // Symmetrise: P_ij = (p(j|i) + p(i|j)) / (2n)
        P = (P + P.transpose()) / (Scalar{2} * static_cast<Scalar>(n));
        P = P.cwiseMax(Scalar{1e-12});
        return P;
    }

    /// @brief Compute the KL divergence D_KL(P || Q) for the final embedding.
    static Scalar compute_kl_divergence(
        const MatrixType& P,
        const MatrixType& Y,
        Eigen::Index n) {
        MatrixType D_Y = pairwise_squared_distances(Y);
        MatrixType num = (MatrixType::Ones(n, n) + D_Y).cwiseInverse();
        num.diagonal().setZero();

        Scalar sum_num = num.sum();
        if (sum_num < Scalar{1e-12}) sum_num = Scalar{1e-12};
        MatrixType Q = num / sum_num;
        Q = Q.cwiseMax(Scalar{1e-12});

        Scalar kl = Scalar{0};
        for (Eigen::Index i = 0; i < n; ++i) {
            for (Eigen::Index j = 0; j < n; ++j) {
                if (i != j && P(i, j) > Scalar{1e-12}) {
                    kl += P(i, j) * std::log(P(i, j) / Q(i, j));
                }
            }
        }
        return kl;
    }
};

/// @}

} // namespace Skigen

#endif // SKIGEN_MANIFOLD_TSNE_H
