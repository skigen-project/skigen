// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_NEURAL_NETWORK_MLP_CLASSIFIER_H
#define SKIGEN_NEURAL_NETWORK_MLP_CLASSIFIER_H

#include "../Core/Base.h"
#include "../Core/Validation.h"
#include "MLPRegressor.h"   // for MLPActivation enum

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

/// @brief Multi-layer perceptron binary classifier (cross-entropy loss).
///
/// Mirrors the binary case of
/// [sklearn.neural_network.MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html).
///
/// Architecture: hidden layers with the chosen activation, a logistic
/// (sigmoid) output unit, trained by mini-batch SGD on the binary
/// cross-entropy loss with L2 weight regularisation (`alpha`).
///
/// ### Limitations relative to scikit-learn
///
/// Binary classification only — multiclass via softmax is not
/// implemented. Only the constant-rate SGD solver is provided; Adam,
/// L-BFGS, early stopping, momentum, and the non-constant learning-rate
/// schedules are not implemented.
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

    MLPClassifier& fit_impl(
        const Eigen::Ref<const MatrixType>& X,
        const Eigen::Ref<const Eigen::VectorXi>& y) {
        internal::check_non_empty(X);
        internal::check_consistent_length(X, y);
        this->n_features_in_ = X.cols();

        // Discover sorted unique labels (binary only).
        std::vector<int> uniq;
        uniq.reserve(static_cast<std::size_t>(y.size()));
        for (Eigen::Index i = 0; i < y.size(); ++i) uniq.push_back(y(i));
        std::sort(uniq.begin(), uniq.end());
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
        if (uniq.size() != 2) {
            throw std::invalid_argument(
                "MLPClassifier: only binary classification is "
                "supported; got " + std::to_string(uniq.size()) +
                " classes.");
        }
        classes_ = Eigen::VectorXi(2);
        classes_(0) = uniq[0];
        classes_(1) = uniq[1];

        // Encode targets as 0/1 against classes_(1) being positive.
        const Eigen::Index n = X.rows();
        VectorType y01(n);
        for (Eigen::Index i = 0; i < n; ++i) {
            y01(i) = (y(i) == classes_(1)) ? Scalar{1} : Scalar{0};
        }

        // Build layer sizes [n_features, h1, h2, ..., 1].
        std::vector<int> sizes;
        sizes.push_back(static_cast<int>(X.cols()));
        for (int h : hidden_layer_sizes_) sizes.push_back(h);
        sizes.push_back(1);

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
                    yb(i) = y01(ii);
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

    [[nodiscard]] LabelType predict_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        VectorType prob = predict_proba_pos(X);
        LabelType out(X.rows());
        for (Eigen::Index i = 0; i < X.rows(); ++i) {
            out(i) = (prob(i) >= Scalar{0.5}) ? classes_(1) : classes_(0);
        }
        return out;
    }

    /// @brief Probability matrix shape (n_samples, 2).
    [[nodiscard]] MatrixType predict_proba(
        const Eigen::Ref<const MatrixType>& X) const {
        VectorType pp = predict_proba_pos(X);
        MatrixType P(X.rows(), 2);
        for (Eigen::Index i = 0; i < X.rows(); ++i) {
            P(i, 0) = Scalar{1} - pp(i);
            P(i, 1) = pp(i);
        }
        return P;
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
    [[nodiscard]] VectorType predict_proba_pos(
        const Eigen::Ref<const MatrixType>& X) const {
        this->check_is_fitted();
        MatrixType A = X;
        for (std::size_t l = 0; l < W_.size(); ++l) {
            MatrixType Z = (A * W_[l]).rowwise() + b_[l].transpose();
            const bool is_output = (l + 1 == W_.size());
            if (is_output) {
                VectorType pp(Z.rows());
                for (Eigen::Index i = 0; i < Z.rows(); ++i) {
                    pp(i) = sigmoid_scalar(Z(i, 0));
                }
                return pp;
            }
            A = activate(Z);
        }
        return VectorType{};
    }

    Scalar sgd_step(const Eigen::Ref<const MatrixType>& Xb,
                    const Eigen::Ref<const VectorType>& yb) {
        const Eigen::Index m = Xb.rows();
        const std::size_t L = W_.size();
        std::vector<MatrixType> As(L);
        MatrixType A = Xb;
        for (std::size_t l = 0; l < L; ++l) {
            MatrixType Z = (A * W_[l]).rowwise() + b_[l].transpose();
            const bool is_output = (l + 1 == L);
            if (is_output) {
                MatrixType outA(Z.rows(), 1);
                for (Eigen::Index i = 0; i < Z.rows(); ++i) {
                    outA(i, 0) = sigmoid_scalar(Z(i, 0));
                }
                A = std::move(outA);
            } else {
                A = activate(Z);
            }
            As[l] = A;
        }
        // Cross-entropy loss on the final sigmoid output.
        Scalar bce{0};
        for (Eigen::Index i = 0; i < m; ++i) {
            const Scalar p = std::clamp(
                As.back()(i, 0),
                std::numeric_limits<Scalar>::epsilon(),
                Scalar{1} - std::numeric_limits<Scalar>::epsilon());
            bce += -(yb(i) * std::log(p) +
                     (Scalar{1} - yb(i)) * std::log(Scalar{1} - p));
        }
        // Backprop: for sigmoid + BCE the output-layer delta simplifies
        // to (p - y) / m.
        MatrixType delta(m, 1);
        for (Eigen::Index i = 0; i < m; ++i) {
            delta(i, 0) =
                (As.back()(i, 0) - yb(i)) / static_cast<Scalar>(m);
        }
        const MatrixType X_in = Xb;
        for (std::size_t l = L; l-- > 0;) {
            const MatrixType& A_prev = (l == 0) ? X_in : As[l - 1];
            MatrixType grad_W = A_prev.transpose() * delta;
            grad_W += alpha_ * W_[l];
            VectorType grad_b = delta.colwise().sum();
            W_[l] -= learning_rate_init_ * grad_W;
            b_[l] -= learning_rate_init_ * grad_b;
            if (l > 0) {
                MatrixType d_prev = delta * W_[l].transpose();
                d_prev = d_prev.cwiseProduct(activate_grad(As[l - 1]));
                delta = std::move(d_prev);
            }
        }
        return bce;
    }

    std::vector<int> hidden_layer_sizes_;
    MLPActivation activation_;
    Scalar alpha_;
    Scalar learning_rate_init_;
    int max_iter_;
    Scalar tol_;
    int batch_size_;
    std::optional<uint64_t> random_state_;

    Eigen::VectorXi classes_;
    std::vector<MatrixType> W_;
    std::vector<VectorType> b_;
    int    n_iter_{0};
    Scalar loss_{0};
};

/// @}

}  // namespace Skigen

#endif  // SKIGEN_NEURAL_NETWORK_MLP_CLASSIFIER_H
