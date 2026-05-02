// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_LINEAR_MODEL_SGD_H
#define SKIGEN_LINEAR_MODEL_SGD_H

#include "../Core/Validation.h"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace Skigen {

/// @defgroup Algo_SGD SGD (Stochastic Gradient Descent)
/// @ingroup LinearModels
/// @brief Linear classifiers and regressors trained via Stochastic Gradient Descent.
/// @{

/// @brief Linear classifier fitted by minimizing a regularized empirical loss with SGD.
///
/// SGDClassifier implements a plain Stochastic Gradient Descent learning
/// routine that supports hinge loss (linear SVM) and log loss (logistic
/// regression). Binary classification uses a single weight vector;
/// multiclass is handled via one-vs-rest.
///
/// Mirrors
/// [sklearn.linear_model.SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html).
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `loss` | `Loss` | `Loss::Hinge` | The loss function to use: `Loss::Hinge` gives a linear SVM, `Loss::Log` gives logistic regression. |
/// | `alpha` | `Scalar` | `1e-4` | Constant that multiplies the regularization term (L2). |
/// | `max_iter` | `int` | `1000` | Maximum number of passes over the training data (epochs). |
/// | `tol` | `Scalar` | `1e-3` | The stopping criterion: training stops when average update per sample is below `tol`. |
/// | `eta0` | `Scalar` | `0.01` | The initial learning rate. |
/// | `random_state` | `unsigned int` | `42` | Seed for the random number generator used for shuffling. |
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type | Description |
/// |----------|------|-------------|
/// | `coef()` | `MatrixType` | Coefficient matrix (n_classes × n_features or 1 × n_features). |
/// | `intercept()` | `VectorType` | Intercept (bias) vector of shape (n_classes,) or (1,). |
///
/// ### See also
///
/// - Skigen::LogisticRegression — Logistic regression with IRLS solver (typically more accurate for small datasets).
///
/// ### Notes
///
/// The learning rate schedule is inverse scaling: @f$\eta_t = \eta_0 / (1 + \eta_0 \alpha t)@f$.
///
/// @note **scikit-learn parity gaps:** The following sklearn constructor
///   parameters are not yet supported: `penalty` (only L2), `l1_ratio`,
///   `fit_intercept`, `shuffle` (always on), `epsilon`, `n_jobs`,
///   `learning_rate` (only inverse scaling), `power_t`, `early_stopping`,
///   `validation_fraction`, `n_iter_no_change`, `class_weight`,
///   `warm_start`, `average`.
///   The following sklearn fitted attributes are not yet exposed:
///   `classes_`, `n_iter_`, `t_`, `n_features_in_`, `feature_names_in_`,
///   `loss_function_`.
///
/// ### Examples
///
/// @snippet sgd.cpp example_sgd_classifier
template <typename Scalar = double>
class SGDClassifier {
public:
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using RowVectorType = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;
    using IndexVector = Eigen::VectorXi;

    /// Loss function type.
    enum class Loss { Hinge, Log };

    /// @brief Construct an SGDClassifier.
    ///
    /// @param loss The loss function (`Loss::Hinge` or `Loss::Log`, default `Loss::Hinge`).
    /// @param alpha Regularization constant (`Scalar`, default `1e-4`).
    /// @param max_iter Maximum number of epochs (`int`, default `1000`).
    /// @param tol Stopping tolerance (`Scalar`, default `1e-3`).
    /// @param eta0 Initial learning rate (`Scalar`, default `0.01`).
    /// @param random_state RNG seed (`unsigned int`, default `42`).
    explicit SGDClassifier(Loss loss = Loss::Hinge, Scalar alpha = Scalar{1e-4},
                           int max_iter = 1000, Scalar tol = Scalar{1e-3},
                           Scalar eta0 = Scalar{0.01},
                           unsigned int random_state = 42)
        : loss_(loss), alpha_(alpha), max_iter_(max_iter), tol_(tol),
          eta0_(eta0), random_state_(random_state) {}

    /// @brief Whether the estimator has been fitted.
    [[nodiscard]] bool is_fitted() const noexcept { return fitted_; }

    /// @brief Coefficient matrix (n_classes × n_features or 1 × n_features).
    ///
    /// @return Read-only reference to the coefficient matrix.
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] const MatrixType& coef() const {
        if (!fitted_) throw std::runtime_error("SGDClassifier not fitted.");
        return coef_;
    }
    /// @brief Intercept (bias) vector of shape (n_classes,) or (1,).
    ///
    /// @return Read-only reference to the intercept vector.
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] const VectorType& intercept() const {
        if (!fitted_) throw std::runtime_error("SGDClassifier not fitted.");
        return intercept_;
    }

    /// @brief Fit the linear model with SGD.
    ///
    /// Discovers unique classes in `y`, then trains a binary classifier
    /// per class (OvR) using stochastic gradient descent with the
    /// chosen loss function.
    ///
    /// @param X Training matrix of shape (n_samples, n_features).
    /// @param y Target vector of shape (n_samples,) with integer class labels.
    /// @return Reference to the fitted estimator (`*this`).
    /// @throws std::invalid_argument if X and y have inconsistent lengths.
    SGDClassifier& fit(const Eigen::Ref<const MatrixType>& X,
                       const Eigen::Ref<const IndexVector>& y) {
        internal::check_non_empty(X);
        if (X.rows() != y.rows()) {
            throw std::invalid_argument("X and y have inconsistent lengths.");
        }

        n_features_in_ = X.cols();
        const Eigen::Index n = X.rows();
        const Eigen::Index p = X.cols();

        // Discover classes
        std::map<int, int> class_map;
        for (Eigen::Index i = 0; i < y.size(); ++i) class_map.emplace(y(i), 0);
        classes_.clear();
        int idx = 0;
        for (auto& [cls, id] : class_map) { id = idx++; classes_.push_back(cls); }

        const int n_classes = static_cast<int>(classes_.size());

        if (n_classes == 2) {
            coef_.resize(1, p);
            intercept_.resize(1);

            VectorType binary_y(n);
            for (Eigen::Index i = 0; i < n; ++i) {
                binary_y(i) = (class_map[y(i)] == 1) ? Scalar{1} : Scalar{-1};
            }

            RowVectorType w;
            Scalar b;
            fit_binary(X, binary_y, w, b);
            coef_.row(0) = w;
            intercept_(0) = b;
        } else {
            // One-vs-Rest
            coef_.resize(n_classes, p);
            intercept_.resize(n_classes);

            for (int c = 0; c < n_classes; ++c) {
                VectorType binary_y(n);
                for (Eigen::Index i = 0; i < n; ++i) {
                    binary_y(i) = (class_map[y(i)] == c) ? Scalar{1} : Scalar{-1};
                }

                RowVectorType w;
                Scalar b;
                fit_binary(X, binary_y, w, b);
                coef_.row(c) = w;
                intercept_(c) = b;
            }
        }

        fitted_ = true;
        return *this;
    }

    /// @brief Predict class labels for samples in X.
    ///
    /// @param X Sample matrix of shape (n_samples, n_features).
    /// @return Integer vector of predicted class labels (n_samples,).
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] IndexVector predict(
        const Eigen::Ref<const MatrixType>& X) const {
        if (!fitted_) throw std::runtime_error("SGDClassifier not fitted.");

        MatrixType scores = X * coef_.transpose();
        scores.rowwise() += intercept_.transpose();

        IndexVector predictions(X.rows());
        if (classes_.size() == 2) {
            for (Eigen::Index i = 0; i < X.rows(); ++i) {
                predictions(i) = scores(i, 0) >= Scalar{0}
                    ? classes_[1] : classes_[0];
            }
        } else {
            for (Eigen::Index i = 0; i < X.rows(); ++i) {
                Eigen::Index max_idx;
                scores.row(i).maxCoeff(&max_idx);
                predictions(i) = classes_[static_cast<std::size_t>(max_idx)];
            }
        }

        return predictions;
    }

    /// @brief Return the mean accuracy on the given test data and labels.
    ///
    /// @param X Test samples of shape (n_samples, n_features).
    /// @param y True class labels of shape (n_samples,).
    /// @return Mean accuracy (fraction of correctly classified samples).
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
    Loss loss_;
    Scalar alpha_;
    int max_iter_;
    Scalar tol_;
    Scalar eta0_;
    unsigned int random_state_;

    bool fitted_ = false;
    Eigen::Index n_features_in_ = 0;
    MatrixType coef_;
    VectorType intercept_;
    std::vector<int> classes_;

    static Scalar sigmoid(Scalar x) {
        if (x >= Scalar{0}) {
            Scalar z = std::exp(-x);
            return Scalar{1} / (Scalar{1} + z);
        }
        Scalar z = std::exp(x);
        return z / (Scalar{1} + z);
    }

    void fit_binary(const Eigen::Ref<const MatrixType>& X,
                    const VectorType& y,
                    RowVectorType& w_out, Scalar& b_out) const {
        const Eigen::Index n = X.rows();
        const Eigen::Index p = X.cols();

        RowVectorType w = RowVectorType::Zero(p);
        Scalar b{0};
        std::mt19937 rng(random_state_);

        std::vector<Eigen::Index> indices(static_cast<std::size_t>(n));
        std::iota(indices.begin(), indices.end(), Eigen::Index{0});

        for (int epoch = 0; epoch < max_iter_; ++epoch) {
            std::shuffle(indices.begin(), indices.end(), rng);
            Scalar eta = eta0_ / (Scalar{1} + eta0_ * alpha_ * static_cast<Scalar>(epoch));
            Scalar total_update{0};

            for (auto idx : indices) {
                Scalar margin = (X.row(idx) * w.transpose())(0) + b;
                Scalar yi = y(idx);

                if (loss_ == Loss::Hinge) {
                    if (yi * margin < Scalar{1}) {
                        w += eta * (yi * X.row(idx) - alpha_ * w);
                        b += eta * yi;
                        total_update += std::abs(eta * yi);
                    } else {
                        w *= (Scalar{1} - eta * alpha_);
                    }
                } else { // Log loss
                    Scalar p_val = sigmoid(yi * margin);
                    Scalar coeff = (Scalar{1} - p_val) * yi;
                    w += eta * (coeff * X.row(idx) - alpha_ * w);
                    b += eta * coeff;
                    total_update += std::abs(eta * coeff);
                }
            }

            if (total_update / static_cast<Scalar>(n) < tol_) break;
        }

        w_out = w;
        b_out = b;
    }
};

/// @brief Linear regressor fitted by minimizing a regularized empirical loss with SGD.
///
/// SGDRegressor implements regularized linear regression with
/// stochastic gradient descent (squared error loss, L2 penalty).
///
/// Mirrors
/// [sklearn.linear_model.SGDRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html).
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `alpha` | `Scalar` | `1e-4` | Constant that multiplies the regularization term (L2). |
/// | `max_iter` | `int` | `1000` | Maximum number of passes over the training data (epochs). |
/// | `tol` | `Scalar` | `1e-3` | The stopping criterion: training stops when average update per sample is below `tol`. |
/// | `eta0` | `Scalar` | `0.01` | The initial learning rate. |
/// | `random_state` | `unsigned int` | `42` | Seed for the random number generator used for shuffling. |
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type | Description |
/// |----------|------|-------------|
/// | `coef()` | `RowVectorType` | Parameter vector @f$w@f$ of shape (1 × n_features). |
/// | `intercept()` | `Scalar` | Independent term in the decision function. |
///
/// ### Notes
///
/// The learning rate schedule is inverse scaling: @f$\eta_t = \eta_0 / (1 + \eta_0 \alpha t)@f$.
///
/// @note **scikit-learn parity gaps:** The following sklearn constructor
///   parameters are not yet supported: `loss` (only squared error),
///   `penalty` (only L2), `l1_ratio`, `fit_intercept`, `shuffle` (always on),
///   `epsilon`, `learning_rate` (only inverse scaling), `power_t`,
///   `early_stopping`, `validation_fraction`, `n_iter_no_change`,
///   `warm_start`, `average`.
///   The following sklearn fitted attributes are not yet exposed:
///   `n_iter_`, `t_`, `n_features_in_`, `feature_names_in_`.
///
/// ### Examples
///
/// @snippet sgd.cpp example_sgd_regressor
template <typename Scalar = double>
class SGDRegressor {
public:
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using RowVectorType = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;

    /// @brief Construct an SGDRegressor.
    ///
    /// @param alpha Regularization constant (`Scalar`, default `1e-4`).
    /// @param max_iter Maximum number of epochs (`int`, default `1000`).
    /// @param tol Stopping tolerance (`Scalar`, default `1e-3`).
    /// @param eta0 Initial learning rate (`Scalar`, default `0.01`).
    /// @param random_state RNG seed (`unsigned int`, default `42`).
    explicit SGDRegressor(Scalar alpha = Scalar{1e-4},
                          int max_iter = 1000, Scalar tol = Scalar{1e-3},
                          Scalar eta0 = Scalar{0.01},
                          unsigned int random_state = 42)
        : alpha_(alpha), max_iter_(max_iter), tol_(tol),
          eta0_(eta0), random_state_(random_state) {}

    /// @brief Whether the estimator has been fitted.
    [[nodiscard]] bool is_fitted() const noexcept { return fitted_; }

    /// @brief Parameter vector @f$w@f$ (1 × n_features).
    ///
    /// @return Read-only reference to the coefficient row-vector.
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] const RowVectorType& coef() const {
        if (!fitted_) throw std::runtime_error("SGDRegressor not fitted.");
        return coef_;
    }
    /// @brief Independent term in the decision function.
    ///
    /// @return The intercept value.
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] Scalar intercept() const {
        if (!fitted_) throw std::runtime_error("SGDRegressor not fitted.");
        return intercept_;
    }

    /// @brief Fit the linear model with SGD.
    ///
    /// @param X Training matrix of shape (n_samples, n_features).
    /// @param y Target vector of shape (n_samples,).
    /// @return Reference to the fitted estimator (`*this`).
    /// @throws std::invalid_argument if X and y have inconsistent lengths.
    SGDRegressor& fit(const Eigen::Ref<const MatrixType>& X,
                      const Eigen::Ref<const VectorType>& y) {
        internal::check_non_empty(X);
        if (X.rows() != y.rows()) {
            throw std::invalid_argument("X and y have inconsistent lengths.");
        }

        n_features_in_ = X.cols();
        const Eigen::Index n = X.rows();
        const Eigen::Index p = X.cols();

        coef_ = RowVectorType::Zero(p);
        intercept_ = Scalar{0};
        std::mt19937 rng(random_state_);

        std::vector<Eigen::Index> indices(static_cast<std::size_t>(n));
        std::iota(indices.begin(), indices.end(), Eigen::Index{0});

        for (int epoch = 0; epoch < max_iter_; ++epoch) {
            std::shuffle(indices.begin(), indices.end(), rng);
            Scalar eta = eta0_ / (Scalar{1} + eta0_ * alpha_ * static_cast<Scalar>(epoch));
            Scalar total_update{0};

            for (auto idx : indices) {
                Scalar pred = (X.row(idx) * coef_.transpose())(0) + intercept_;
                Scalar error = pred - y(idx);

                coef_ -= eta * (error * X.row(idx) + alpha_ * coef_);
                intercept_ -= eta * error;
                total_update += std::abs(eta * error);
            }

            if (total_update / static_cast<Scalar>(n) < tol_) break;
        }

        fitted_ = true;
        return *this;
    }

    /// @brief Predict using the linear model.
    ///
    /// @param X Sample matrix of shape (n_samples, n_features).
    /// @return Predicted values of shape (n_samples,).
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] VectorType predict(
        const Eigen::Ref<const MatrixType>& X) const {
        if (!fitted_) throw std::runtime_error("SGDRegressor not fitted.");
        return (X * coef_.transpose()).array() + intercept_;
    }

    /// @brief Return the @f$R^2@f$ coefficient of determination on test data.
    ///
    /// @param X Test samples of shape (n_samples, n_features).
    /// @param y True values of shape (n_samples,).
    /// @return @f$R^2@f$ score.
    [[nodiscard]] Scalar score(const Eigen::Ref<const MatrixType>& X,
                               const Eigen::Ref<const VectorType>& y) const {
        VectorType y_pred = predict(X);
        Scalar ss_res = (y - y_pred).squaredNorm();
        Scalar ss_tot = (y.array() - y.mean()).matrix().squaredNorm();
        if (ss_tot == Scalar{0}) return Scalar{0};
        return Scalar{1} - ss_res / ss_tot;
    }

private:
    Scalar alpha_;
    int max_iter_;
    Scalar tol_;
    Scalar eta0_;
    unsigned int random_state_;

    bool fitted_ = false;
    Eigen::Index n_features_in_ = 0;
    RowVectorType coef_;
    Scalar intercept_{0};
};

/// @}

} // namespace Skigen

#endif // SKIGEN_LINEAR_MODEL_SGD_H
