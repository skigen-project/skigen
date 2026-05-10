// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_NEURAL_NETWORK_MLP_REGRESSOR_H
#define SKIGEN_NEURAL_NETWORK_MLP_REGRESSOR_H

#include "../Core/Base.h"
#include "../Core/Validation.h"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace Skigen {

/// @defgroup Algo_MLP Multi-Layer Perceptron
/// @ingroup NeuralNetwork
/// @brief Dense feed-forward neural networks trained by mini-batch SGD.
/// @{

/// @brief MLP activation choices (matches sklearn's `activation=` strings).
enum class MLPActivation { Identity, Logistic, Tanh, ReLU };

/// @brief Multi-layer perceptron regressor with squared-error loss.
///
/// Mirrors
/// [sklearn.neural_network.MLPRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html).
///
/// Architecture: an arbitrary stack of fully-connected hidden layers with
/// the chosen activation, followed by a linear output unit. Trained by
/// mini-batch SGD on the squared-error loss with L2 weight regularisation
/// (`alpha`). Single-target only.
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default |
/// |---|---|---|
/// | `hidden_layer_sizes`    | `vector<int>`         | `{100}` |
/// | `activation`            | `MLPActivation`       | `ReLU` |
/// | `alpha`                 | `Scalar`              | `1e-4` |
/// | `learning_rate_init`    | `Scalar`              | `1e-3` |
/// | `max_iter`              | `int`                 | `200` |
/// | `tol`                   | `Scalar`              | `1e-4` |
/// | `batch_size`            | `int`                 | `min(200, n_samples)` *(0 = auto)* |
/// | `random_state`          | `optional<uint64_t>`  | `nullopt` |
///
/// ### Limitations relative to scikit-learn
///
/// Only the constant-rate SGD solver is implemented; Adam and L-BFGS,
/// momentum / Nesterov, the inverse-scaling and adaptive learning-rate
/// schedules, early stopping with `validation_fraction` /
/// `n_iter_no_change`, and the Adam moment hyperparameters
/// (`beta_1`, `beta_2`, `epsilon`) are not implemented. Multi-target
/// output is also not supported (the output layer is a single unit).
template <typename Scalar = double>
class MLPRegressor : public Predictor<MLPRegressor<Scalar>, Scalar> {
public:
    using Base = Predictor<MLPRegressor<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;
    using typename Base::ScalarType;

    explicit MLPRegressor(
        std::vector<int> hidden_layer_sizes = {100},
        MLPActivation activation = MLPActivation::ReLU,
        Scalar alpha = Scalar{1e-4},
        Scalar learning_rate_init = Scalar{1e-3},
        int max_iter = 200,
        Scalar tol = Scalar{1e-4},
        int batch_size = 0,
        std::optional<uint64_t> random_state = std::nullopt)
        : hidden_layer_sizes_(std::move(hidden_layer_sizes)),
          activation_(activation),
          alpha_(alpha),
          learning_rate_init_(learning_rate_init),
          max_iter_(max_iter),
          tol_(tol),
          batch_size_(batch_size),
          random_state_(random_state) {}

    [[nodiscard]] const std::vector<int>& hidden_layer_sizes() const noexcept {
        return hidden_layer_sizes_;
    }
    [[nodiscard]] MLPActivation activation() const noexcept {
        return activation_;
    }
    [[nodiscard]] int n_iter_run() const {
        this->check_is_fitted(); return n_iter_;
    }
    [[nodiscard]] Scalar loss() const {
        this->check_is_fitted(); return loss_;
    }
    [[nodiscard]] const std::vector<MatrixType>& coefs() const {
        this->check_is_fitted(); return W_;
    }
    [[nodiscard]] const std::vector<VectorType>& intercepts() const {
        this->check_is_fitted(); return b_;
    }

    MLPRegressor& fit_impl(const Eigen::Ref<const MatrixType>& X,
                           const Eigen::Ref<const VectorType>& y) {
        internal::check_non_empty(X);
        internal::check_consistent_length(X, y);
        this->n_features_in_ = X.cols();

        // Build layer sizes [n_features, h1, h2, ..., 1].
        std::vector<int> sizes;
        sizes.push_back(static_cast<int>(X.cols()));
        for (int h : hidden_layer_sizes_) sizes.push_back(h);
        sizes.push_back(1);

        // Initialise weights with Glorot uniform.
        const uint64_t seed =
            random_state_.value_or(static_cast<uint64_t>(0));
        std::mt19937_64 rng(seed);
        W_.clear();
        b_.clear();
        for (std::size_t l = 1; l < sizes.size(); ++l) {
            const int fan_in  = sizes[l - 1];
            const int fan_out = sizes[l];
            const Scalar limit =
                std::sqrt(Scalar{6} /
                          static_cast<Scalar>(fan_in + fan_out));
            std::uniform_real_distribution<Scalar> d(-limit, limit);
            MatrixType Wl(fan_in, fan_out);
            for (int i = 0; i < fan_in; ++i)
                for (int j = 0; j < fan_out; ++j) Wl(i, j) = d(rng);
            VectorType bl = VectorType::Zero(fan_out);
            W_.push_back(std::move(Wl));
            b_.push_back(std::move(bl));
        }

        const Eigen::Index n = X.rows();
        int bs = (batch_size_ > 0)
            ? std::min(batch_size_, static_cast<int>(n))
            : std::min(200, static_cast<int>(n));

        std::vector<Eigen::Index> indices(static_cast<std::size_t>(n));
        std::iota(indices.begin(), indices.end(), Eigen::Index{0});

        Scalar prev_loss =
            std::numeric_limits<Scalar>::infinity();
        n_iter_ = 0;
        for (int epoch = 0; epoch < max_iter_; ++epoch) {
            std::shuffle(indices.begin(), indices.end(), rng);
            Scalar epoch_loss{0};
            for (Eigen::Index start = 0; start < n; start += bs) {
                const Eigen::Index end =
                    std::min<Eigen::Index>(start + bs, n);
                const Eigen::Index m = end - start;
                MatrixType Xb(m, X.cols());
                VectorType yb(m);
                for (Eigen::Index i = 0; i < m; ++i) {
                    const Eigen::Index ii =
                        indices[static_cast<std::size_t>(start + i)];
                    Xb.row(i) = X.row(ii);
                    yb(i) = y(ii);
                }
                epoch_loss += sgd_step(Xb, yb);
            }
            n_iter_ = epoch + 1;
            const Scalar avg_loss = epoch_loss /
                static_cast<Scalar>(n);
            loss_ = avg_loss;
            if (std::abs(prev_loss - avg_loss) < tol_) break;
            prev_loss = avg_loss;
        }

        this->fitted_ = true;
        return *this;
    }

    [[nodiscard]] VectorType predict_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        MatrixType A = X;
        for (std::size_t l = 0; l < W_.size(); ++l) {
            MatrixType Z = (A * W_[l]).rowwise() + b_[l].transpose();
            const bool is_output = (l + 1 == W_.size());
            A = is_output ? Z : activate(Z);
        }
        return A.col(0);
    }

    [[nodiscard]] Scalar score_impl(
        const Eigen::Ref<const MatrixType>& X,
        const Eigen::Ref<const VectorType>& y) const {
        VectorType yh = predict_impl(X);
        const Scalar ym = y.mean();
        const Scalar ss_res = (y - yh).squaredNorm();
        const Scalar ss_tot =
            (y.array() - ym).matrix().squaredNorm();
        if (ss_tot == Scalar{0}) return Scalar{0};
        return Scalar{1} - ss_res / ss_tot;
    }

private:
    MatrixType activate(const MatrixType& Z) const {
        switch (activation_) {
            case MLPActivation::Identity: return Z;
            case MLPActivation::Logistic:
                return (Scalar{1} /
                        (Scalar{1} + (-Z).array().exp())).matrix();
            case MLPActivation::Tanh:
                return Z.array().tanh().matrix();
            case MLPActivation::ReLU:
                return Z.cwiseMax(Scalar{0});
        }
        return Z;
    }
    MatrixType activate_grad(const MatrixType& A) const {
        switch (activation_) {
            case MLPActivation::Identity:
                return MatrixType::Ones(A.rows(), A.cols());
            case MLPActivation::Logistic:
                return (A.array() * (Scalar{1} - A.array())).matrix();
            case MLPActivation::Tanh:
                return (Scalar{1} - A.array().square()).matrix();
            case MLPActivation::ReLU:
                return (A.array() > Scalar{0})
                    .template cast<Scalar>()
                    .matrix();
        }
        return A;
    }

    Scalar sgd_step(const Eigen::Ref<const MatrixType>& Xb,
                    const Eigen::Ref<const VectorType>& yb) {
        const Eigen::Index m = Xb.rows();
        const std::size_t L = W_.size();
        std::vector<MatrixType> Zs(L), As(L);
        MatrixType A = Xb;
        for (std::size_t l = 0; l < L; ++l) {
            MatrixType Z = (A * W_[l]).rowwise() + b_[l].transpose();
            Zs[l] = Z;
            const bool is_output = (l + 1 == L);
            A = is_output ? Z : activate(Z);
            As[l] = A;
        }
        // Loss = (1/2m) * sum (A_last - y)^2 + alpha/2 * ||W||^2
        VectorType yhat = As.back().col(0);
        VectorType resid = yhat - yb;
        const Scalar mse = Scalar{0.5} * resid.squaredNorm();

        // Backprop: delta_L = (A_L - y) / m  for the linear output layer.
        MatrixType delta(m, 1);
        delta.col(0) = resid / static_cast<Scalar>(m);
        // Materialise Xb as a MatrixType once so the ternary below has
        // matching types on both branches.
        const MatrixType X_in = Xb;
        for (std::size_t l = L; l-- > 0;) {
            const MatrixType& A_prev = (l == 0) ? X_in : As[l - 1];
            MatrixType grad_W = A_prev.transpose() * delta;
            grad_W += alpha_ * W_[l];
            VectorType grad_b = delta.colwise().sum();
            // Step.
            W_[l] -= learning_rate_init_ * grad_W;
            b_[l] -= learning_rate_init_ * grad_b;
            if (l > 0) {
                MatrixType d_prev = delta * W_[l].transpose();
                d_prev = d_prev.cwiseProduct(activate_grad(As[l - 1]));
                delta = std::move(d_prev);
            }
        }
        return mse;
    }

    std::vector<int> hidden_layer_sizes_;
    MLPActivation activation_;
    Scalar alpha_;
    Scalar learning_rate_init_;
    int max_iter_;
    Scalar tol_;
    int batch_size_;
    std::optional<uint64_t> random_state_;

    std::vector<MatrixType> W_;
    std::vector<VectorType> b_;
    int    n_iter_{0};
    Scalar loss_{0};
};

/// @}

}  // namespace Skigen

#endif  // SKIGEN_NEURAL_NETWORK_MLP_REGRESSOR_H
