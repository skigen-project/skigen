// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_LINEAR_MODEL_SGD_H
#define SKIGEN_LINEAR_MODEL_SGD_H

#include "../Core/Base.h"
#include "../Core/Validation.h"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <random>
#include <vector>

namespace Skigen {

/// @defgroup Algo_SGD SGD (Stochastic Gradient Descent)
/// @ingroup LinearModels
/// @brief Linear classifiers and regressors trained via Stochastic Gradient Descent.
/// @{

/// @brief Linear classifier fitted by minimizing a regularized empirical loss with SGD.
///
/// SGDClassifier implements a plain Stochastic Gradient Descent learning
/// routine that supports hinge loss (linear SVM), log loss (logistic
/// regression), and perceptron loss. Binary classification uses a single weight vector;
/// multiclass is handled via one-vs-rest.
///
/// Mirrors
/// [sklearn.linear_model.SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html).
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `loss` | `Loss` | `Loss::Hinge` | The loss function to use: `Loss::Hinge` gives a linear SVM, `Loss::Log` gives logistic regression, `Loss::Perceptron` gives the perceptron update. |
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
/// ### Limitations relative to scikit-learn
///
/// The following scikit-learn constructor
///   parameters are not honoured: `penalty` (only L2), `l1_ratio`,
///   `fit_intercept`, `shuffle` (always on), `epsilon`, `n_jobs`,
///   `learning_rate` (only inverse scaling), `power_t`, `early_stopping`,
///   `validation_fraction`, `n_iter_no_change`, `class_weight`,
///   `warm_start`, `average`.
///   The following sklearn fitted attributes are not exposed:
///   `classes_`, `n_iter_`, `t_`, `n_features_in_`, `feature_names_in_`,
///   `loss_function_`.
///
/// ### Examples
///
/// @snippet sgd.cpp example_sgd_classifier
template <typename Scalar = double>
class SGDClassifier
    : public Classifier<SGDClassifier<Scalar>, Scalar> {
public:
    using Base = Classifier<SGDClassifier<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;
    using typename Base::ScalarType;
    using typename Base::LabelType;
    using IndexVector = Eigen::VectorXi;

    /// Loss function type.
    enum class Loss { Hinge, Log, Perceptron };

    /// @brief Construct an SGDClassifier.
    ///
    /// @param loss The loss function (`Loss::Hinge`, `Loss::Log`, or `Loss::Perceptron`, default `Loss::Hinge`).
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

    /// @brief Coefficient matrix (n_classes × n_features or 1 × n_features).
    [[nodiscard]] const MatrixType& coef() const {
        this->check_is_fitted();
        return coef_;
    }
    /// @brief Intercept (bias) vector of shape (n_classes,) or (1,).
    [[nodiscard]] const VectorType& intercept() const {
        this->check_is_fitted();
        return intercept_;
    }

    SKIGEN_PARAMS(
        (alpha,    alpha_,    double),
        (max_iter, max_iter_, int),
        (tol,      tol_,      double),
        (eta0,     eta0_,     double))

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
    SGDClassifier& fit_impl(const Eigen::Ref<const MatrixType>& X,
                       const Eigen::Ref<const IndexVector>& y) {
        internal::check_non_empty(X);
        internal::check_consistent_length(X, y);

        this->n_features_in_ = X.cols();
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

        this->fitted_ = true;
        return *this;
    }

    /// @brief Predict class labels for samples in X.
    [[nodiscard]] IndexVector predict_impl(
        const Eigen::Ref<const MatrixType>& X) const {

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

private:
    Loss loss_;
    Scalar alpha_;
    int max_iter_;
    Scalar tol_;
    Scalar eta0_;
    unsigned int random_state_;

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
        // Cold-start full fit: initialise from zero and run up to max_iter
        // epochs with the early-stop tolerance.
        const Eigen::Index p = X.cols();
        RowVectorType w = RowVectorType::Zero(p);
        Scalar b{0};
        run_sgd_epochs(X, y, w, b, /*n_epochs=*/max_iter_,
                       /*epoch0=*/0, /*use_tol=*/true);
        w_out = w;
        b_out = b;
    }

    // Run `n_epochs` of SGD on (X, y) starting from current (w, b).
    // `epoch0` parameterises the inverse-scaling learning rate so resumed
    // calls in partial_fit() decay consistently with the cold-start path.
    // When `use_tol` is true, the loop early-stops on the average-update
    // criterion (used by `fit`); for partial_fit we set it to false so a
    // single mini-epoch always runs to completion.
    void run_sgd_epochs(const Eigen::Ref<const MatrixType>& X,
                        const VectorType& y,
                        RowVectorType& w, Scalar& b,
                        int n_epochs, int epoch0, bool use_tol) const {
        const Eigen::Index n = X.rows();
        std::mt19937 rng(random_state_ +
                         static_cast<unsigned>(epoch0));
        std::vector<Eigen::Index> indices(static_cast<std::size_t>(n));
        std::iota(indices.begin(), indices.end(), Eigen::Index{0});

        for (int e = 0; e < n_epochs; ++e) {
            const int epoch = epoch0 + e;
            std::shuffle(indices.begin(), indices.end(), rng);
            const Scalar eta = eta0_ /
                (Scalar{1} + eta0_ * alpha_ * static_cast<Scalar>(epoch));
            Scalar total_update{0};

            for (auto idx : indices) {
                const Scalar margin = (X.row(idx) * w.transpose())(0) + b;
                const Scalar yi = y(idx);

                if (loss_ == Loss::Hinge) {
                    if (yi * margin < Scalar{1}) {
                        w += eta * (yi * X.row(idx) - alpha_ * w);
                        b += eta * yi;
                        total_update += std::abs(eta * yi);
                    } else {
                        w *= (Scalar{1} - eta * alpha_);
                    }
                } else if (loss_ == Loss::Perceptron) {
                    if (yi * margin <= Scalar{0}) {
                        w += eta * (yi * X.row(idx) - alpha_ * w);
                        b += eta * yi;
                        total_update += std::abs(eta * yi);
                    } else {
                        w *= (Scalar{1} - eta * alpha_);
                    }
                } else { // Log loss
                    const Scalar p_val = sigmoid(yi * margin);
                    const Scalar coeff = (Scalar{1} - p_val) * yi;
                    w += eta * (coeff * X.row(idx) - alpha_ * w);
                    b += eta * coeff;
                    total_update += std::abs(eta * coeff);
                }
            }
            if (use_tol &&
                total_update / static_cast<Scalar>(n) < tol_) break;
        }
    }

public:
    // -- partial_fit ---------------------------------------------------------

    /// @brief Online SGD update.
    ///
    /// Runs **one epoch** of SGD on the supplied (X, y) batch starting from
    /// the current `coef_` / `intercept_` (matching sklearn's
    /// `SGDClassifier.partial_fit` contract).
    ///
    /// The first call requires `classes` (an Eigen::VectorXi of all
    /// possible labels — a sklearn convention for the
    /// `classes` argument); subsequent calls accept an empty
    /// `classes` vector and reuse the already-discovered class set.
    ///
    /// @throws std::invalid_argument when the feature count differs from
    ///   the first batch, or when the first call omits `classes`.
    SGDClassifier& partial_fit(
        const Eigen::Ref<const MatrixType>& X,
        const Eigen::Ref<const IndexVector>& y,
        const Eigen::Ref<const IndexVector>& classes) {
        internal::check_non_empty(X);
        internal::check_consistent_length(X, y);

        if (!this->fitted_) {
            if (classes.size() < 2) {
                throw std::invalid_argument(
                    "partial_fit: the `classes` argument is required on "
                    "the first call and must contain >= 2 classes.");
            }
            classes_.clear();
            classes_.reserve(static_cast<std::size_t>(classes.size()));
            for (Eigen::Index i = 0; i < classes.size(); ++i) {
                classes_.push_back(classes(i));
            }
            std::sort(classes_.begin(), classes_.end());
            classes_.erase(std::unique(classes_.begin(), classes_.end()),
                           classes_.end());
            this->n_features_in_ = X.cols();
            const int K = static_cast<int>(classes_.size());
            const Eigen::Index p = X.cols();
            if (K == 2) {
                coef_ = MatrixType::Zero(1, p);
                intercept_ = VectorType::Zero(1);
            } else {
                coef_ = MatrixType::Zero(K, p);
                intercept_ = VectorType::Zero(K);
            }
            n_iter_partial_ = 0;
            this->fitted_ = true;
        } else {
            if (X.cols() != this->n_features_in_) {
                throw std::invalid_argument(
                    "X has " + std::to_string(X.cols()) + " features, but "
                    "partial_fit was previously called with " +
                    std::to_string(this->n_features_in_) + " features.");
            }
        }

        // Build label-to-index map from already-discovered classes_.
        std::map<int, int> class_map;
        for (std::size_t i = 0; i < classes_.size(); ++i) {
            class_map[classes_[i]] = static_cast<int>(i);
        }

        const Eigen::Index n = X.rows();
        const int K = static_cast<int>(classes_.size());

        if (K == 2) {
            VectorType binary_y(n);
            for (Eigen::Index i = 0; i < n; ++i) {
                auto it = class_map.find(y(i));
                if (it == class_map.end()) {
                    throw std::invalid_argument(
                        "partial_fit: encountered a label not in classes_.");
                }
                binary_y(i) = (it->second == 1) ? Scalar{1} : Scalar{-1};
            }
            RowVectorType w = coef_.row(0);
            Scalar b = intercept_(0);
            run_sgd_epochs(X, binary_y, w, b, /*n_epochs=*/1,
                           /*epoch0=*/n_iter_partial_,
                           /*use_tol=*/false);
            coef_.row(0) = w;
            intercept_(0) = b;
        } else {
            for (int c = 0; c < K; ++c) {
                VectorType binary_y(n);
                for (Eigen::Index i = 0; i < n; ++i) {
                    auto it = class_map.find(y(i));
                    if (it == class_map.end()) {
                        throw std::invalid_argument(
                            "partial_fit: encountered a label not in classes_.");
                    }
                    binary_y(i) = (it->second == c) ? Scalar{1} : Scalar{-1};
                }
                RowVectorType w = coef_.row(c);
                Scalar b = intercept_(c);
                run_sgd_epochs(X, binary_y, w, b, /*n_epochs=*/1,
                               /*epoch0=*/n_iter_partial_,
                               /*use_tol=*/false);
                coef_.row(c) = w;
                intercept_(c) = b;
            }
        }
        ++n_iter_partial_;
        return *this;
    }

private:
    int n_iter_partial_ = 0;
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
/// ### Limitations relative to scikit-learn
///
/// The following scikit-learn constructor
///   parameters are not honoured: `loss` (only squared error),
///   `penalty` (only L2), `l1_ratio`, `fit_intercept`, `shuffle` (always on),
///   `epsilon`, `learning_rate` (only inverse scaling), `power_t`,
///   `early_stopping`, `validation_fraction`, `n_iter_no_change`,
///   `warm_start`, `average`.
///   The following sklearn fitted attributes are not exposed:
///   `n_iter_`, `t_`, `n_features_in_`, `feature_names_in_`.
///
/// ### Examples
///
/// @snippet sgd.cpp example_sgd_regressor
template <typename Scalar = double>
class SGDRegressor
    : public Predictor<SGDRegressor<Scalar>, Scalar> {
public:
    using Base = Predictor<SGDRegressor<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;
    using typename Base::ScalarType;

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

    /// @brief Parameter vector @f$w@f$ (1 × n_features).
    [[nodiscard]] const RowVectorType& coef() const {
        this->check_is_fitted();
        return coef_;
    }
    /// @brief Independent term in the decision function.
    [[nodiscard]] Scalar intercept_value() const {
        this->check_is_fitted();
        return intercept_;
    }

    SKIGEN_PARAMS(
        (alpha,    alpha_,    double),
        (max_iter, max_iter_, int),
        (tol,      tol_,      double),
        (eta0,     eta0_,     double))

    /// @brief Fit the linear model with SGD.
    ///
    /// @param X Training matrix of shape (n_samples, n_features).
    /// @param y Target vector of shape (n_samples,).
    /// @return Reference to the fitted estimator (`*this`).
    /// @throws std::invalid_argument if X and y have inconsistent lengths.
    SGDRegressor& fit_impl(const Eigen::Ref<const MatrixType>& X,
                      const Eigen::Ref<const VectorType>& y) {
        internal::check_non_empty(X);
        internal::check_consistent_length(X, y);

        this->n_features_in_ = X.cols();
        const Eigen::Index p = X.cols();

        coef_ = RowVectorType::Zero(p);
        intercept_ = Scalar{0};
        n_iter_partial_ = 0;
        run_epochs(X, y, /*n_epochs=*/max_iter_,
                   /*epoch0=*/0, /*use_tol=*/true);

        this->fitted_ = true;
        return *this;
    }

    /// @brief Online SGD update — runs one epoch on (X, y) starting from
    ///   the current `coef_` / `intercept_`, matching sklearn's
    ///   `SGDRegressor.partial_fit` contract. The first call initialises
    ///   the state to zero (no `classes` argument is needed for
    ///   regression). Throws on feature-count mismatch with the first
    ///   batch.
    SGDRegressor& partial_fit(
        const Eigen::Ref<const MatrixType>& X,
        const Eigen::Ref<const VectorType>& y) {
        internal::check_non_empty(X);
        internal::check_consistent_length(X, y);
        if (!this->fitted_) {
            this->n_features_in_ = X.cols();
            coef_ = RowVectorType::Zero(X.cols());
            intercept_ = Scalar{0};
            n_iter_partial_ = 0;
            this->fitted_ = true;
        } else if (X.cols() != this->n_features_in_) {
            throw std::invalid_argument(
                "X has " + std::to_string(X.cols()) + " features, but "
                "partial_fit was previously called with " +
                std::to_string(this->n_features_in_) + " features.");
        }
        run_epochs(X, y, /*n_epochs=*/1,
                   /*epoch0=*/n_iter_partial_,
                   /*use_tol=*/false);
        ++n_iter_partial_;
        return *this;
    }

private:
    void run_epochs(const Eigen::Ref<const MatrixType>& X,
                    const Eigen::Ref<const VectorType>& y,
                    int n_epochs, int epoch0, bool use_tol) {
        const Eigen::Index n = X.rows();
        std::mt19937 rng(random_state_ +
                         static_cast<unsigned>(epoch0));
        std::vector<Eigen::Index> indices(static_cast<std::size_t>(n));
        std::iota(indices.begin(), indices.end(), Eigen::Index{0});

        for (int e = 0; e < n_epochs; ++e) {
            const int epoch = epoch0 + e;
            std::shuffle(indices.begin(), indices.end(), rng);
            const Scalar eta = eta0_ /
                (Scalar{1} + eta0_ * alpha_ * static_cast<Scalar>(epoch));
            Scalar total_update{0};
            for (auto idx : indices) {
                const Scalar pred =
                    (X.row(idx) * coef_.transpose())(0) + intercept_;
                const Scalar error = pred - y(idx);
                coef_      -= eta * (error * X.row(idx) + alpha_ * coef_);
                intercept_ -= eta * error;
                total_update += std::abs(eta * error);
            }
            if (use_tol &&
                total_update / static_cast<Scalar>(n) < tol_) break;
        }
    }
public:

    /// @brief Predict using the linear model.
    [[nodiscard]] VectorType predict_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        return (X * coef_.transpose()).array() + intercept_;
    }

    /// @brief Return the @f$R^2@f$ coefficient of determination on test data.
    [[nodiscard]] ScalarType score_impl(const Eigen::Ref<const MatrixType>& X,
                               const Eigen::Ref<const VectorType>& y) const {
        VectorType y_pred = predict_impl(X);
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

    RowVectorType coef_;
    Scalar intercept_{0};
    int n_iter_partial_ = 0;
};

/// @}

} // namespace Skigen

#endif // SKIGEN_LINEAR_MODEL_SGD_H
