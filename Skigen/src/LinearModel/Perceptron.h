// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

#ifndef SKIGEN_LINEAR_MODEL_PERCEPTRON_H
#define SKIGEN_LINEAR_MODEL_PERCEPTRON_H

#include "SGD.h"

namespace Skigen {

/// @defgroup Algo_Perceptron Perceptron
/// @ingroup LinearModels
/// @brief Linear perceptron classifier.
/// @{

/// @brief Linear perceptron classifier.
///
/// Thin compatibility estimator configured as stochastic gradient descent with
/// perceptron loss, constant initial step size, and no regularization by default.
///
/// Mirrors the dense core of
/// [sklearn.linear_model.Perceptron](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html).
///
/// ### Examples
///
/// @snippet perceptron.cpp example_perceptron
template <typename Scalar = double>
class Perceptron : public Classifier<Perceptron<Scalar>, Scalar> {
public:
    using Base = Classifier<Perceptron<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::IndexType;
    using typename Base::LabelType;

    explicit Perceptron(int max_iter = 1000,
                        Scalar tol = Scalar{1e-3},
                        Scalar eta0 = Scalar{1},
                        unsigned int random_state = 42)
        : model_(SGDClassifier<Scalar>::Loss::Perceptron,
                 /*alpha=*/Scalar{0},
                 max_iter,
                 tol,
                 eta0,
                 random_state),
          max_iter_(max_iter),
          tol_(tol),
          eta0_(eta0),
          random_state_(random_state) {}

    [[nodiscard]] int max_iter() const noexcept { return max_iter_; }
    [[nodiscard]] Scalar tol() const noexcept { return tol_; }
    [[nodiscard]] Scalar eta0() const noexcept { return eta0_; }
    [[nodiscard]] unsigned int random_state() const noexcept { return random_state_; }

    [[nodiscard]] const MatrixType& coef() const {
        this->check_is_fitted();
        return model_.coef();
    }

    [[nodiscard]] const VectorType& intercept() const {
        this->check_is_fitted();
        return model_.intercept();
    }

    Perceptron& fit_impl(const Eigen::Ref<const MatrixType>& X,
                         const Eigen::Ref<const LabelType>& y) {
        model_.fit(X, y);
        this->n_features_in_ = X.cols();
        this->fitted_ = true;
        return *this;
    }

    [[nodiscard]] LabelType predict_impl(const Eigen::Ref<const MatrixType>& X) const {
        return model_.predict(X);
    }

    [[nodiscard]] Scalar score_impl(const Eigen::Ref<const MatrixType>& X,
                                    const Eigen::Ref<const LabelType>& y) const {
        return model_.score(X, y);
    }

    Perceptron& partial_fit(const Eigen::Ref<const MatrixType>& X,
                            const Eigen::Ref<const LabelType>& y,
                            const Eigen::Ref<const LabelType>& classes) {
        model_.partial_fit(X, y, classes);
        this->n_features_in_ = X.cols();
        this->fitted_ = true;
        return *this;
    }

private:
    SGDClassifier<Scalar> model_;
    int max_iter_;
    Scalar tol_;
    Scalar eta0_;
    unsigned int random_state_;
};

/// @}

}  // namespace Skigen

#endif  // SKIGEN_LINEAR_MODEL_PERCEPTRON_H
