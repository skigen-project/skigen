// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_LINEAR_MODEL_RIDGE_H
#define SKIGEN_LINEAR_MODEL_RIDGE_H

#include "../Core/Base.h"
#include "../Core/Validation.h"

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/SparseCore>

namespace Skigen {

/// @defgroup Algo_Ridge Ridge Regression
/// @ingroup LinearModels
/// @brief L2-regularized least squares (Ridge regression).
/// @see https://skigen-project.github.io/docs/guide/ridge for algorithm intuition.
/// @{

/// @brief Linear least squares with L2 regularization.
///
/// Minimizes the objective function:
///
/// @f[
///   \|y - Xw\|_2^2 + \alpha \|w\|_2^2
/// @f]
///
/// This model solves a regression model where the loss function is the
/// linear least squares function and regularization is given by the
/// L2-norm. Also known as Ridge Regression or Tikhonov regularization.
///
/// Mirrors
/// [sklearn.linear_model.Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html).
///
/// Read more in the @ref guide_ridge "User Guide".
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `alpha` | `Scalar` | `1` | Constant that multiplies the L2 term, controlling regularization strength. `alpha = 0` is equivalent to ordinary least squares (use LinearRegression instead). |
/// | `fit_intercept` | `bool` | `true` | Whether the intercept should be estimated. If `false`, the data is assumed to be already centered. |
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type | Description |
/// |----------|------|-------------|
/// | `coef()` | `RowVectorType` | Parameter vector @f$w@f$ of shape (1 × n_features). |
/// | `intercept()` | `Scalar` | Independent term in the decision function. |
///
/// ### See also
///
/// - Skigen::LinearRegression — Ordinary least squares without regularization.
/// - Skigen::Lasso — L1-only regularization (coordinate descent).
/// - Skigen::ElasticNet — Combined L1 + L2 regularization.
///
/// ### Notes
///
/// Uses the Cholesky-based normal equation
/// @f$(X^\top X + \alpha I)w = X^\top y@f$.
///
/// Regularization improves the conditioning of the problem and reduces
/// the variance of the estimates. Larger values specify stronger
/// regularization.
///
/// ### Limitations relative to scikit-learn
///
/// Only the Cholesky solver is implemented; `solver` selection,
/// `max_iter`, `tol`, `positive`, `random_state`, and `sample_weight`
/// are not honoured. The fitted attributes `n_iter_`,
/// `feature_names_in_`, and `solver_` are not exposed.
///
/// ### Examples
///
/// @snippet ridge.cpp example_ridge
template <typename Scalar = double>
class Ridge
    : public Predictor<Ridge<Scalar>, Scalar> {
public:
    using Base = Predictor<Ridge<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;
    using typename Base::ScalarType;

    // Make the dense base-class fit overload visible alongside the
    // sparse fit overload added below.
    using Base::fit;

    // Register parameters with the reflection layer. The macro generates
    // `set_param_impl` and `get_params_impl` hooks that the Estimator
    // base's `set_param` / `get_params` dispatch into; hyperparameter-
    // search drivers like GridSearchCV use these to read and write
    // settings by name.
    SKIGEN_PARAMS(
        (alpha,         alpha_,         double),
        (fit_intercept, fit_intercept_, bool))

    /// @brief Construct a Ridge estimator.
    ///
    /// @param alpha Constant that multiplies the L2 term (`Scalar`, default `1`).
    ///   Controls regularization strength. `alpha = 0` is equivalent to
    ///   ordinary least squares (use LinearRegression instead for numerical
    ///   stability).
    /// @param fit_intercept Whether the intercept should be estimated (`bool`, default `true`).
    ///   If `false`, the data is assumed to be already centered.
    explicit Ridge(Scalar alpha = Scalar{1}, bool fit_intercept = true)
        : alpha_(alpha), fit_intercept_(fit_intercept) {}

    // -- Accessors ----------------------------------------------------------

    /// @brief Regularization strength.
    [[nodiscard]] Scalar alpha() const noexcept { return alpha_; }
    /// @brief Whether an intercept is fitted.
    [[nodiscard]] bool fit_intercept() const noexcept { return fit_intercept_; }

    /// @brief Parameter vector @f$w@f$ (1 × n_features).
    ///
    /// Estimated coefficients for the linear regression problem.
    /// @return Read-only reference to the coefficient row-vector.
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] const RowVectorType& coef() const {
        this->check_is_fitted(); return coef_;
    }
    /// @brief Independent term in the decision function.
    ///
    /// Set to `0` if `fit_intercept = false`.
    /// @return The intercept value.
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] Scalar intercept() const {
        this->check_is_fitted(); return intercept_;
    }

    // -- Implementation (called by CRTP base) --------------------------------

    /// @brief Fit the Ridge model via Cholesky decomposition.
    ///
    /// Centers the data when `fit_intercept` is `true`, then solves
    /// the regularized normal equation via Cholesky factorization.
    ///
    /// @param X Design matrix of shape (n_samples, n_features).
    /// @param y Target vector of shape (n_samples,).
    ///   Will be cast to `Scalar` if necessary.
    /// @return Reference to the fitted estimator (`*this`).
    ///
    /// `sample_weight` is not honoured.
    Ridge& fit_impl(const Eigen::Ref<const MatrixType>& X,
                    const Eigen::Ref<const VectorType>& y) {
        internal::check_non_empty(X);
        internal::check_consistent_length(X, y);

        this->n_features_in_ = X.cols();
        // Reset multi-target view so it's lazily re-synthesised next access.
        coef_matrix_.resize(0, 0);
        intercept_vector_.resize(0);
        n_targets_ = 1;

        MatrixType X_c;
        VectorType y_c;
        Scalar y_offset{0};

        if (fit_intercept_) {
            x_offset_ = X.colwise().mean();
            y_offset = y.mean();
            X_c = X.rowwise() - x_offset_;
            y_c = y.array() - y_offset;
        } else {
            X_c = X;
            y_c = y;
        }

        // (X^T X + αI) w = X^T y — Cholesky solver
        MatrixType XtX = X_c.transpose() * X_c;
        XtX.diagonal().array() += alpha_;

        VectorType Xty = X_c.transpose() * y_c;

        Eigen::LLT<MatrixType> llt(XtX);
        VectorType w = llt.solve(Xty);

        coef_ = w.transpose();

        if (fit_intercept_) {
            intercept_ = y_offset - x_offset_.dot(w);
        } else {
            intercept_ = Scalar{0};
        }

        this->fitted_ = true;
        return *this;
    }

    // -- Sparse-aware overload ----------------------------------------------

    /// @brief Fit Ridge on a sparse design matrix without densifying X.
    ///
    /// Solves the regularised normal equation
    /// @f$ (X_c^\top X_c + \alpha I) w = X_c^\top y_c @f$ where the
    /// centring is applied implicitly via the identities
    ///
    /// @f[
    ///   X_c^\top X_c = X^\top X - n\, \bar{x}^\top \bar{x},\qquad
    ///   X_c^\top y_c = X^\top y - n\, \bar{y}\, \bar{x}^\top
    /// @f]
    ///
    /// so that `X` itself is never centered (which would densify it).
    /// `X^\top X` (`p \times p`) and `X^\top y` are computed directly on
    /// the sparse representation; only the small Gram matrix is
    /// materialised dense, then factored with Cholesky.
    ///
    /// `sample_weight`, `solver` selection, `tol`, `max_iter`, and
    /// `positive` are not honoured on the sparse path either.
    template <int Options, typename StorageIndex>
    Ridge& fit(const Eigen::SparseMatrix<Scalar, Options, StorageIndex>& X,
               const Eigen::Ref<const VectorType>& y) {
        if (X.rows() == 0 || X.cols() == 0) {
            throw std::invalid_argument("Ridge.fit: empty sparse matrix.");
        }
        if (X.rows() != y.size()) {
            throw std::invalid_argument(
                "Ridge.fit: X has " + std::to_string(X.rows()) +
                " rows but y has " + std::to_string(y.size()) + ".");
        }
        this->n_features_in_ = X.cols();

        using ColSparse =
            Eigen::SparseMatrix<Scalar, Eigen::ColMajor, StorageIndex>;
        const ColSparse Xc = X;
        const Eigen::Index n = Xc.rows();
        const Eigen::Index p = Xc.cols();

        // Per-column means computed directly from the CSC nonzeros.
        RowVectorType x_mean = RowVectorType::Zero(p);
        if (fit_intercept_) {
            for (Eigen::Index j = 0; j < p; ++j) {
                Scalar s{0};
                for (typename ColSparse::InnerIterator it(Xc, j); it; ++it) {
                    s += it.value();
                }
                x_mean(j) = s / static_cast<Scalar>(n);
            }
        }
        const Scalar y_mean = fit_intercept_ ? y.mean() : Scalar{0};

        // X^T X (sparse * sparse densified into a small p×p dense matrix).
        MatrixType XtX = MatrixType(Xc.transpose() * Xc);
        VectorType Xty = Xc.transpose() * y;

        if (fit_intercept_) {
            // Implicit centring: X_c^T X_c = X^T X - n * x_mean^T x_mean.
            XtX.noalias() -= static_cast<Scalar>(n) *
                             (x_mean.transpose() * x_mean);
            // X_c^T y_c = X^T y - n * y_mean * x_mean^T.
            Xty.noalias() -= static_cast<Scalar>(n) * y_mean *
                             x_mean.transpose();
        }

        XtX.diagonal().array() += alpha_;

        Eigen::LLT<MatrixType> llt(XtX);
        VectorType w = llt.solve(Xty);

        coef_ = w.transpose();
        if (fit_intercept_) {
            intercept_ = y_mean - x_mean.dot(w);
            x_offset_ = x_mean;
        } else {
            intercept_ = Scalar{0};
        }

        this->fitted_ = true;
        return *this;
    }

    /// @brief Predict using the linear model.
    ///
    /// Computes @f$ \hat{y} = X w + b @f$ where @f$w@f$ and @f$b@f$ are
    /// the fitted coefficients and intercept.
    ///
    /// @param X Sample matrix of shape (n_samples, n_features).
    /// @return Predicted values of shape (n_samples,).
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] VectorType predict_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        return (X * coef_.transpose()).array() + intercept_;
    }

    /// @brief Return the @f$R^2@f$ coefficient of determination on test data.
    ///
    /// @f$ R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2} @f$.
    /// Best possible score is 1.0; it can be negative if the model is
    /// arbitrarily worse than predicting the mean.
    ///
    /// @param X Test samples of shape (n_samples, n_features).
    /// @param y True values of shape (n_samples,).
    /// @return @f$R^2@f$ score.
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] ScalarType score_impl(
        const Eigen::Ref<const MatrixType>& X,
        const Eigen::Ref<const VectorType>& y) const {
        internal::check_consistent_length(X, y);
        VectorType y_pred = predict_impl(X);
        Scalar ss_res = (y - y_pred).squaredNorm();
        Scalar ss_tot = (y.array() - y.mean()).matrix().squaredNorm();
        if (ss_tot == Scalar{0}) return Scalar{0};
        return Scalar{1} - ss_res / ss_tot;
    }

    // -- Multi-target regression --------------------------------------------

    /// @brief Fit Ridge with a multi-target response matrix.
    ///
    /// Solves @f$ (X_c^\top X_c + \alpha I) W = X_c^\top Y_c @f$ via a single
    /// Cholesky factorisation shared across targets — `LLT::solve` accepts
    /// a matrix RHS, so per-target cost is just the back-substitution.
    /// Coefficient shape is (n_targets, n_features); intercept shape is
    /// (n_targets,).
    ///
    /// Additive: the single-target API (`coef()` / `intercept()`) keeps
    /// working and reflects the first target column. New accessors:
    /// `coef_matrix()`, `intercept_vector()`, `n_targets()`,
    /// `predict_multi(X)`.
    Ridge& fit_multi(const Eigen::Ref<const MatrixType>& X,
                     const Eigen::Ref<const MatrixType>& Y) {
        internal::check_non_empty(X);
        if (X.rows() != Y.rows()) {
            throw std::invalid_argument(
                "fit_multi: X has " + std::to_string(X.rows()) +
                " rows but Y has " + std::to_string(Y.rows()) + ".");
        }
        if (Y.cols() < 1) {
            throw std::invalid_argument(
                "fit_multi: Y must have at least 1 target column.");
        }
        this->n_features_in_ = X.cols();

        MatrixType X_c;
        MatrixType Y_c;
        RowVectorType y_offset = RowVectorType::Zero(Y.cols());

        if (fit_intercept_) {
            x_offset_ = X.colwise().mean();
            y_offset  = Y.colwise().mean();
            X_c = X.rowwise() - x_offset_;
            Y_c = Y.rowwise() - y_offset;
        } else {
            X_c = X;
            Y_c = Y;
        }

        MatrixType XtX = X_c.transpose() * X_c;
        XtX.diagonal().array() += alpha_;
        MatrixType XtY = X_c.transpose() * Y_c;

        Eigen::LLT<MatrixType> llt(XtX);
        MatrixType W = llt.solve(XtY);              // (n_features, n_targets)

        coef_matrix_ = W.transpose();               // (n_targets, n_features)
        if (fit_intercept_) {
            intercept_vector_ = y_offset.transpose() -
                                coef_matrix_ * x_offset_.transpose();
        } else {
            intercept_vector_ = VectorType::Zero(Y.cols());
        }

        // Mirror first target into the single-target accessors.
        coef_      = coef_matrix_.row(0);
        intercept_ = intercept_vector_(0);
        n_targets_ = static_cast<int>(Y.cols());

        this->fitted_ = true;
        return *this;
    }

    /// @brief Coefficient matrix of shape (n_targets, n_features).
    [[nodiscard]] const MatrixType& coef_matrix() const {
        this->check_is_fitted();
        if (coef_matrix_.size() == 0) {
            coef_matrix_.resize(1, coef_.size());
            coef_matrix_.row(0) = coef_;
            intercept_vector_.resize(1);
            intercept_vector_(0) = intercept_;
            n_targets_ = 1;
        }
        return coef_matrix_;
    }
    /// @brief Intercept vector of length n_targets.
    [[nodiscard]] const VectorType& intercept_vector() const {
        this->check_is_fitted();
        if (intercept_vector_.size() == 0) {
            intercept_vector_.resize(1);
            intercept_vector_(0) = intercept_;
            coef_matrix_.resize(1, coef_.size());
            coef_matrix_.row(0) = coef_;
            n_targets_ = 1;
        }
        return intercept_vector_;
    }
    [[nodiscard]] int n_targets() const {
        this->check_is_fitted(); return n_targets_;
    }
    /// @brief Multi-target predict, shape (n_samples, n_targets).
    [[nodiscard]] MatrixType predict_multi(
        const Eigen::Ref<const MatrixType>& X) const {
        this->check_is_fitted();
        this->validate_feature_count(X);
        const MatrixType& W = coef_matrix();
        const VectorType& b = intercept_vector();
        MatrixType Y_pred = X * W.transpose();
        Y_pred.rowwise() += b.transpose();
        return Y_pred;
    }

private:
    Scalar alpha_;
    bool fit_intercept_;

    RowVectorType coef_;
    Scalar intercept_ = Scalar{0};
    RowVectorType x_offset_;

    // Multi-target state (lazy 1-target view for the single-target fit).
    mutable MatrixType  coef_matrix_;
    mutable VectorType  intercept_vector_;
    mutable int         n_targets_ = 0;
};

/// @}

} // namespace Skigen

#endif // SKIGEN_LINEAR_MODEL_RIDGE_H
