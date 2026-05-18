// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_NEURAL_NETWORK_MLP_CLASSIFIER_H
#define SKIGEN_NEURAL_NETWORK_MLP_CLASSIFIER_H

#include "../Core/Base.h"
#include "../Core/Validation.h"
#include "MLPRegressor.h"   // for MLPActivation, MLPSolver enums

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace Skigen {

/// @addtogroup Algo_MLP
/// @{

/// @brief Multi-layer perceptron classifier (binary + multiclass).
///
/// Mirrors
/// [sklearn.neural_network.MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html).
///
/// Architecture: hidden layers with the chosen activation, followed by
/// a logistic (sigmoid) output unit for binary or softmax for multiclass.
/// Trained by mini-batch SGD or Adam on the cross-entropy loss with L2
/// weight regularisation (`alpha`).
template <typename Scalar = double>
class MLPClassifier : public Classifier<MLPClassifier<Scalar>, Scalar> {
public:
    using Base = Classifier<MLPClassifier<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;
    using typename Base::ScalarType;
    using typename Base::LabelType;
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    explicit MLPClassifier(
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
                "MLPClassifier: solver='lbfgs' is not implemented.");
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
    [[nodiscard]] const Eigen::VectorXi& classes() const {
        this->check_is_fitted(); return classes_;
    }
    [[nodiscard]] int n_classes() const {
        this->check_is_fitted(); return static_cast<int>(classes_.size());
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

    SKIGEN_PARAMS((alpha, alpha_, double),
                  (learning_rate_init, learning_rate_init_, double),
                  (max_iter, max_iter_, int),
                  (tol, tol_, double),
                  (batch_size, batch_size_, int),
                  (beta_1, beta_1_, double),
                  (beta_2, beta_2_, double),
                  (epsilon, epsilon_, double))

    MLPClassifier& fit_impl(
        const Eigen::Ref<const MatrixType>& X,
        const Eigen::Ref<const Eigen::VectorXi>& y) {
        internal::check_non_empty(X);
        internal::check_consistent_length(X, y);
        this->n_features_in_ = X.cols();

        std::vector<int> uniq;
        uniq.reserve(static_cast<std::size_t>(y.size()));
        for (Eigen::Index i = 0; i < y.size(); ++i) uniq.push_back(y(i));
        std::sort(uniq.begin(), uniq.end());
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
        if (uniq.size() < 2) {
            throw std::invalid_argument(
                "MLPClassifier: need at least 2 classes; got " +
                std::to_string(uniq.size()) + ".");
        }
        const int K = static_cast<int>(uniq.size());
        classes_.resize(K);
        for (int c = 0; c < K; ++c) classes_(c) = uniq[static_cast<std::size_t>(c)];

        const Eigen::Index n = X.rows();
        const int n_out = (K == 2) ? 1 : K;

        MatrixType Y_enc(n, n_out);
        if (K == 2) {
            for (Eigen::Index i = 0; i < n; ++i)
                Y_enc(i, 0) = (y(i) == classes_(1)) ? Scalar{1} : Scalar{0};
        } else {
            Y_enc.setZero();
            for (Eigen::Index i = 0; i < n; ++i) {
                for (int c = 0; c < K; ++c) {
                    if (y(i) == classes_(c)) { Y_enc(i, c) = Scalar{1}; break; }
                }
            }
        }

        std::vector<int> sizes;
        sizes.push_back(static_cast<int>(X.cols()));
        for (int h : hidden_layer_sizes_) sizes.push_back(h);
        sizes.push_back(n_out);

        const uint64_t seed =
            random_state_.value_or(static_cast<uint64_t>(0));
        std::mt19937_64 rng(seed);
        init_weights(sizes, rng);

        int bs = (batch_size_ > 0)
            ? std::min(batch_size_, static_cast<int>(n))
            : std::min(200, static_cast<int>(n));

        std::vector<Eigen::Index> indices(static_cast<std::size_t>(n));
        std::iota(indices.begin(), indices.end(), Eigen::Index{0});

        Scalar prev_loss = std::numeric_limits<Scalar>::infinity();
        n_iter_ = 0;
        for (int epoch = 0; epoch < max_iter_; ++epoch) {
            std::shuffle(indices.begin(), indices.end(), rng);
            Scalar epoch_loss{0};
            for (Eigen::Index start = 0; start < n; start += bs) {
                const Eigen::Index end =
                    std::min<Eigen::Index>(start + bs, n);
                const Eigen::Index m = end - start;
                MatrixType Xb(m, X.cols());
                MatrixType Yb(m, n_out);
                for (Eigen::Index i = 0; i < m; ++i) {
                    const Eigen::Index ii =
                        indices[static_cast<std::size_t>(start + i)];
                    Xb.row(i) = X.row(ii);
                    Yb.row(i) = Y_enc.row(ii);
                }
                epoch_loss += training_step(Xb, Yb, K);
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

    [[nodiscard]] LabelType predict_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        this->check_is_fitted();
        const int K = static_cast<int>(classes_.size());
        LabelType out(X.rows());
        if (K == 2) {
            VectorType prob = predict_proba_binary(X);
            for (Eigen::Index i = 0; i < X.rows(); ++i)
                out(i) = (prob(i) >= Scalar{0.5}) ? classes_(1) : classes_(0);
        } else {
            MatrixType P = predict_proba_multi(X);
            for (Eigen::Index i = 0; i < X.rows(); ++i) {
                Eigen::Index best;
                P.row(i).maxCoeff(&best);
                out(i) = classes_(static_cast<int>(best));
            }
        }
        return out;
    }

    /// @brief Probability matrix shape (n_samples, n_classes).
    [[nodiscard]] MatrixType predict_proba(
        const Eigen::Ref<const MatrixType>& X) const {
        this->check_is_fitted();
        const int K = static_cast<int>(classes_.size());
        if (K == 2) {
            VectorType pp = predict_proba_binary(X);
            MatrixType P(X.rows(), 2);
            P.col(0) = (Scalar{1} - pp.array()).matrix();
            P.col(1) = pp;
            return P;
        }
        return predict_proba_multi(X);
    }

private:
    static Scalar sigmoid_scalar(Scalar x) {
        if (x >= Scalar{0}) {
            const Scalar z = std::exp(-x);
            return Scalar{1} / (Scalar{1} + z);
        }
        const Scalar z = std::exp(x);
        return z / (Scalar{1} + z);
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

    static MatrixType softmax(const MatrixType& Z) {
        MatrixType P(Z.rows(), Z.cols());
        for (Eigen::Index i = 0; i < Z.rows(); ++i) {
            const Scalar row_max = Z.row(i).maxCoeff();
            Eigen::Matrix<Scalar, 1, Eigen::Dynamic> e =
                (Z.row(i).array() - row_max).exp();
            P.row(i) = e / e.sum();
        }
        return P;
    }

    [[nodiscard]] VectorType predict_proba_binary(
        const Eigen::Ref<const MatrixType>& X) const {
        MatrixType A = X;
        for (std::size_t l = 0; l < W_.size(); ++l) {
            MatrixType Z = (A * W_[l]).rowwise() + b_[l].transpose();
            const bool is_output = (l + 1 == W_.size());
            if (is_output) {
                VectorType pp(Z.rows());
                for (Eigen::Index i = 0; i < Z.rows(); ++i)
                    pp(i) = sigmoid_scalar(Z(i, 0));
                return pp;
            }
            A = activate(Z);
        }
        return VectorType{};
    }

    [[nodiscard]] MatrixType predict_proba_multi(
        const Eigen::Ref<const MatrixType>& X) const {
        MatrixType A = X;
        for (std::size_t l = 0; l < W_.size(); ++l) {
            MatrixType Z = (A * W_[l]).rowwise() + b_[l].transpose();
            const bool is_output = (l + 1 == W_.size());
            if (is_output) return softmax(Z);
            A = activate(Z);
        }
        return MatrixType{};
    }

    void init_weights(const std::vector<int>& sizes,
                      std::mt19937_64& rng) {
        W_.clear(); b_.clear();
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
                         const Eigen::Ref<const MatrixType>& Yb,
                         int K) {
        const Eigen::Index m = Xb.rows();
        const std::size_t L = W_.size();
        std::vector<MatrixType> As(L);
        MatrixType A = Xb;
        for (std::size_t l = 0; l < L; ++l) {
            MatrixType Z = (A * W_[l]).rowwise() + b_[l].transpose();
            const bool is_output = (l + 1 == L);
            if (is_output) {
                if (K == 2) {
                    MatrixType outA(Z.rows(), 1);
                    for (Eigen::Index i = 0; i < Z.rows(); ++i)
                        outA(i, 0) = sigmoid_scalar(Z(i, 0));
                    A = std::move(outA);
                } else {
                    A = softmax(Z);
                }
            } else {
                A = activate(Z);
            }
            As[l] = A;
        }

        const Scalar eps = std::numeric_limits<Scalar>::epsilon();
        Scalar ce{0};
        if (K == 2) {
            for (Eigen::Index i = 0; i < m; ++i) {
                const Scalar p = std::clamp(As.back()(i, 0), eps, Scalar{1} - eps);
                ce += -(Yb(i, 0) * std::log(p) +
                        (Scalar{1} - Yb(i, 0)) * std::log(Scalar{1} - p));
            }
        } else {
            for (Eigen::Index i = 0; i < m; ++i)
                for (int c = 0; c < K; ++c) {
                    const Scalar p = std::clamp(As.back()(i, c), eps, Scalar{1} - eps);
                    ce += -Yb(i, c) * std::log(p);
                }
        }

        MatrixType delta = (As.back() - Yb) / static_cast<Scalar>(m);

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
        return ce;
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

    Eigen::VectorXi classes_;
    std::vector<MatrixType> W_;
    std::vector<VectorType> b_;
    std::vector<MatrixType> m_W_, v_W_;
    std::vector<VectorType> m_b_, v_b_;
    int    adam_t_{0};
    int    n_iter_{0};
    Scalar loss_{0};
};

/// @}

}  // namespace Skigen

#endif  // SKIGEN_NEURAL_NETWORK_MLP_CLASSIFIER_H
