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
/// @brief Dense feed-forward neural networks trained by mini-batch SGD or Adam.
/// @{

/// @brief MLP activation choices (matches sklearn's `activation=` strings).
enum class MLPActivation { Identity, Logistic, Tanh, ReLU };

/// @brief MLP solver choices (matches sklearn's `solver=` strings).
enum class MLPSolver { Adam, SGD, LBFGS };

/// @brief Multi-layer perceptron regressor with squared-error loss.
///
/// Mirrors
/// [sklearn.neural_network.MLPRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html).
///
/// Architecture: an arbitrary stack of fully-connected hidden layers with
/// the chosen activation, followed by a linear output layer. Trained by
/// mini-batch SGD or Adam on the squared-error loss with L2 weight
/// regularisation (`alpha`).
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default |
/// |---|---|---|
/// | `hidden_layer_sizes`    | `vector<int>`         | `{100}` |
/// | `activation`            | `MLPActivation`       | `ReLU` |
/// | `solver`                | `MLPSolver`           | `Adam` |
/// | `alpha`                 | `Scalar`              | `1e-4` |
/// | `learning_rate_init`    | `Scalar`              | `1e-3` |
/// | `max_iter`              | `int`                 | `200` |
/// | `tol`                   | `Scalar`              | `1e-4` |
/// | `batch_size`            | `int`                 | `min(200, n_samples)` *(0 = auto)* |
/// | `random_state`          | `optional<uint64_t>`  | `nullopt` |
/// | `beta_1`                | `Scalar`              | `0.9` |
/// | `beta_2`                | `Scalar`              | `0.999` |
/// | `epsilon`               | `Scalar`              | `1e-8` |
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
        MLPSolver solver = MLPSolver::Adam,
        Scalar alpha = Scalar{1e-4},
        Scalar learning_rate_init = Scalar{1e-3},
        int max_iter = 200,
        Scalar tol = Scalar{1e-4},
        int batch_size = 0,
        std::optional<uint64_t> random_state = std::nullopt,
        Scalar beta_1 = Scalar{0.9},
        Scalar beta_2 = Scalar{0.999},
        Scalar epsilon = Scalar{1e-8})
        : hidden_layer_sizes_(std::move(hidden_layer_sizes)),
          activation_(activation),
          solver_(solver),
          alpha_(alpha),
          learning_rate_init_(learning_rate_init),
          max_iter_(max_iter),
          tol_(tol),
          batch_size_(batch_size),
          random_state_(random_state),
          beta_1_(beta_1),
          beta_2_(beta_2),
          epsilon_(epsilon) {
        if (solver_ == MLPSolver::LBFGS) {
            throw std::invalid_argument(
                "MLPRegressor: solver='lbfgs' is not implemented.");
        }
    }

    [[nodiscard]] const std::vector<int>& hidden_layer_sizes() const noexcept {
        return hidden_layer_sizes_;
    }
    [[nodiscard]] MLPActivation activation() const noexcept {
        return activation_;
    }
    [[nodiscard]] MLPSolver solver() const noexcept {
        return solver_;
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
    [[nodiscard]] int n_outputs() const {
        this->check_is_fitted(); return n_outputs_;
    }

    SKIGEN_PARAMS((alpha, alpha_, double),
                  (learning_rate_init, learning_rate_init_, double),
                  (max_iter, max_iter_, int),
                  (tol, tol_, double),
                  (batch_size, batch_size_, int),
                  (beta_1, beta_1_, double),
                  (beta_2, beta_2_, double),
                  (epsilon, epsilon_, double))

    MLPRegressor& fit_impl(const Eigen::Ref<const MatrixType>& X,
                           const Eigen::Ref<const VectorType>& y) {
        MatrixType Y = y;
        Y.resize(y.size(), 1);
        return fit_internal(X, Y);
    }

    /// @brief Fit on multiple target columns.
    MLPRegressor& fit_multi(const Eigen::Ref<const MatrixType>& X,
                            const Eigen::Ref<const MatrixType>& Y) {
        internal::check_non_empty(X);
        if (X.rows() != Y.rows()) {
            throw std::invalid_argument(
                "MLPRegressor: X and Y must have the same number of rows.");
        }
        return fit_internal(X, Y);
    }

    [[nodiscard]] VectorType predict_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        MatrixType A = forward(X);
        return A.col(0);
    }

    /// @brief Predict multiple targets (returns n_samples × n_targets matrix).
    [[nodiscard]] MatrixType predict_multi(
        const Eigen::Ref<const MatrixType>& X) const {
        this->check_is_fitted();
        return forward(X);
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
    MatrixType forward(const Eigen::Ref<const MatrixType>& X) const {
        MatrixType A = X;
        for (std::size_t l = 0; l < W_.size(); ++l) {
            MatrixType Z = (A * W_[l]).rowwise() + b_[l].transpose();
            const bool is_output = (l + 1 == W_.size());
            A = is_output ? Z : activate(Z);
        }
        return A;
    }

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

    MLPRegressor& fit_internal(const Eigen::Ref<const MatrixType>& X,
                               const Eigen::Ref<const MatrixType>& Y) {
        internal::check_non_empty(X);
        this->n_features_in_ = X.cols();
        n_outputs_ = static_cast<int>(Y.cols());

        std::vector<int> sizes;
        sizes.push_back(static_cast<int>(X.cols()));
        for (int h : hidden_layer_sizes_) sizes.push_back(h);
        sizes.push_back(n_outputs_);

        const uint64_t seed =
            random_state_.value_or(static_cast<uint64_t>(0));
        std::mt19937_64 rng(seed);
        init_weights(sizes, rng);

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
                MatrixType Yb(m, Y.cols());
                for (Eigen::Index i = 0; i < m; ++i) {
                    const Eigen::Index ii =
                        indices[static_cast<std::size_t>(start + i)];
                    Xb.row(i) = X.row(ii);
                    Yb.row(i) = Y.row(ii);
                }
                epoch_loss += training_step(Xb, Yb);
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

    void init_weights(const std::vector<int>& sizes,
                      std::mt19937_64& rng) {
        W_.clear();
        b_.clear();
        m_W_.clear(); v_W_.clear();
        m_b_.clear(); v_b_.clear();
        adam_t_ = 0;
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
            if (solver_ == MLPSolver::Adam) {
                m_W_.push_back(MatrixType::Zero(fan_in, fan_out));
                v_W_.push_back(MatrixType::Zero(fan_in, fan_out));
                m_b_.push_back(VectorType::Zero(fan_out));
                v_b_.push_back(VectorType::Zero(fan_out));
            }
        }
    }

    Scalar training_step(const Eigen::Ref<const MatrixType>& Xb,
                         const Eigen::Ref<const MatrixType>& Yb) {
        const Eigen::Index m = Xb.rows();
        const std::size_t L = W_.size();
        std::vector<MatrixType> As(L);
        MatrixType A = Xb;
        for (std::size_t l = 0; l < L; ++l) {
            MatrixType Z = (A * W_[l]).rowwise() + b_[l].transpose();
            const bool is_output = (l + 1 == L);
            A = is_output ? Z : activate(Z);
            As[l] = A;
        }
        MatrixType resid = As.back() - Yb;
        const Scalar mse = Scalar{0.5} * resid.squaredNorm();

        MatrixType delta = resid / static_cast<Scalar>(m);

        if (solver_ == MLPSolver::Adam) ++adam_t_;

        const MatrixType X_in = Xb;
        for (std::size_t l = L; l-- > 0;) {
            const MatrixType& A_prev = (l == 0) ? X_in : As[l - 1];
            MatrixType grad_W = A_prev.transpose() * delta;
            grad_W += alpha_ * W_[l];
            VectorType grad_b = delta.colwise().sum();

            if (solver_ == MLPSolver::Adam) {
                const Scalar bc1 =
                    Scalar{1} - std::pow(beta_1_, static_cast<Scalar>(adam_t_));
                const Scalar bc2 =
                    Scalar{1} - std::pow(beta_2_, static_cast<Scalar>(adam_t_));
                m_W_[l] = beta_1_ * m_W_[l] + (Scalar{1} - beta_1_) * grad_W;
                v_W_[l] = beta_2_ * v_W_[l] +
                    (Scalar{1} - beta_2_) * grad_W.cwiseProduct(grad_W);
                MatrixType m_hat = m_W_[l] / bc1;
                MatrixType v_hat = v_W_[l] / bc2;
                W_[l] -= learning_rate_init_ *
                    (m_hat.array() /
                     (v_hat.array().sqrt() + epsilon_)).matrix();
                m_b_[l] = beta_1_ * m_b_[l] + (Scalar{1} - beta_1_) * grad_b;
                v_b_[l] = beta_2_ * v_b_[l] +
                    (Scalar{1} - beta_2_) * grad_b.cwiseProduct(grad_b);
                VectorType mb_hat = m_b_[l] / bc1;
                VectorType vb_hat = v_b_[l] / bc2;
                b_[l] -= learning_rate_init_ *
                    (mb_hat.array() /
                     (vb_hat.array().sqrt() + epsilon_)).matrix();
            } else {
                W_[l] -= learning_rate_init_ * grad_W;
                b_[l] -= learning_rate_init_ * grad_b;
            }

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
    MLPSolver solver_;
    Scalar alpha_;
    Scalar learning_rate_init_;
    int max_iter_;
    Scalar tol_;
    int batch_size_;
    std::optional<uint64_t> random_state_;
    Scalar beta_1_;
    Scalar beta_2_;
    Scalar epsilon_;

    std::vector<MatrixType> W_;
    std::vector<VectorType> b_;
    std::vector<MatrixType> m_W_, v_W_;
    std::vector<VectorType> m_b_, v_b_;
    int    adam_t_{0};
    int    n_iter_{0};
    int    n_outputs_{1};
    Scalar loss_{0};
};

/// @}

}  // namespace Skigen

#endif  // SKIGEN_NEURAL_NETWORK_MLP_REGRESSOR_H
