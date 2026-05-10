// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_LINEAR_MODEL_LINEAR_REGRESSION_H
#define SKIGEN_LINEAR_MODEL_LINEAR_REGRESSION_H

#include "../Core/Base.h"
#include "../Core/Validation.h"

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/QR>
#include <Eigen/SparseCore>

namespace Skigen {

/// @defgroup Algo_LinearRegression LinearRegression
/// @ingroup LinearModels
/// @brief Ordinary Least Squares linear regression.
/// @see https://skigen-project.github.io/docs/guide/linear-regression for algorithm intuition.
/// @{

/// @brief Ordinary least squares Linear Regression.
///
/// LinearRegression fits a linear model with coefficients
/// @f$w = (w_1, \ldots, w_p)@f$ to minimize the residual sum of squares
/// between the observed targets in the dataset, and the targets predicted
/// by the linear approximation:
///
/// @f[
///   \hat{w} = \arg\min_w \|Xw - y\|_2^2
/// @f]
///
/// Solves via ColPivHouseholderQR decomposition. When `fit_intercept` is
/// `true`, data is centered before solving.
///
/// Mirrors
/// [sklearn.linear_model.LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html).
///
/// Read more in the @ref guide_linear_regression "User Guide".
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `fit_intercept` | `bool` | `true` | Whether to calculate the intercept for this model. If `false`, no intercept will be used (data is expected to be centered). |
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type | Description |
/// |----------|------|-------------|
/// | `coef()` | `RowVectorType` | Estimated coefficients of shape (1 × n_features). |
/// | `intercept()` | `Scalar` | Independent term in the linear model. Set to `0` if `fit_intercept = false`. |
/// | `rank()` | `IndexType` | Numerical rank of the design matrix X. |
///
/// ### See also
///
/// - Skigen::Ridge — L2 regularization to reduce overfitting.
/// - Skigen::Lasso — L1 regularization for sparse coefficients.
/// - Skigen::ElasticNet — Combined L1 + L2 regularization.
///
/// ### Notes
///
/// From the implementation point of view, this is QR decomposition
/// (Eigen::ColPivHouseholderQR) wrapped as a predictor object.
///
/// @note **scikit-learn parity gaps:** The following sklearn constructor
///   parameters are not yet supported: `copy_X`, `n_jobs`, `positive`, `tol`.
///   The following sklearn fitted attributes are not yet exposed:
///   `singular_`, `n_features_in_`, `feature_names_in_`.
///   `sample_weight` in `fit()` is not yet supported.
///
/// ### Examples
///
/// @snippet linear_regression.cpp example_linear_regression
template <typename Scalar = double>
class LinearRegression
    : public Predictor<LinearRegression<Scalar>, Scalar> {
public:
    using Base = Predictor<LinearRegression<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;
    using typename Base::ScalarType;

    // Make the dense base-class fit overload visible alongside the
    // sparse fit overload added below.
    using Base::fit;

    /// @brief Construct a LinearRegression estimator.
    ///
    /// @param fit_intercept Whether to calculate the intercept (`bool`, default `true`).
    ///   If `false`, no intercept will be used (data is expected to be centered).
    explicit LinearRegression(bool fit_intercept = true)
        : fit_intercept_(fit_intercept) {}

    // -- Accessors ----------------------------------------------------------

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
    /// @brief Numerical rank of the design matrix X.
    [[nodiscard]] IndexType rank() const {
        this->check_is_fitted(); return rank_;
    }

    // -- Implementation (called by CRTP base) --------------------------------

    /// @brief Fit the model using Ordinary Least Squares.
    ///
    /// Uses ColPivHouseholderQR decomposition. Centers data when
    /// `fit_intercept` is `true`.
    ///
    /// @param X Design matrix of shape (n_samples, n_features).
    /// @param y Target vector of shape (n_samples,).
    ///   Will be cast to `Scalar` if necessary.
    /// @return Reference to the fitted estimator (`*this`).
    ///
    /// @note **sklearn parity gap:** `sample_weight` parameter is
    ///   not yet supported.
    LinearRegression& fit_impl(const Eigen::Ref<const MatrixType>& X,
                               const Eigen::Ref<const VectorType>& y) {
        internal::check_non_empty(X);
        internal::check_consistent_length(X, y);

        this->n_features_in_ = X.cols();
        // Reset multi-target view so the next coef_matrix() / predict_multi()
        // call re-synthesises a fresh 1-target view from coef_/intercept_.
        coef_matrix_.resize(0, 0);
        intercept_vector_.resize(0);
        n_targets_ = 1;

        if (fit_intercept_) {
            x_offset_ = X.colwise().mean();
            const Scalar y_offset = y.mean();

            MatrixType X_c = X.rowwise() - x_offset_;
            VectorType y_c = y.array() - y_offset;

            auto qr = X_c.colPivHouseholderQr();
            coef_ = qr.solve(y_c).transpose();
            rank_ = qr.rank();

            intercept_ = y_offset - x_offset_.dot(coef_.transpose());
        } else {
            auto qr = X.colPivHouseholderQr();
            coef_ = qr.solve(y).transpose();
            rank_ = qr.rank();
            intercept_ = Scalar{0};
        }

        this->fitted_ = true;
        return *this;
    }

    // -- Sparse-aware overload (v1.1.0 §3.2) --------------------------------

    /// @brief Fit OLS on a sparse design matrix without densifying X.
    ///
    /// Solves the (centred when `fit_intercept=true`) normal equation
    /// @f$ (X_c^\top X_c) w = X_c^\top y_c @f$ via the implicit-centring
    /// identities used by `Ridge::fit(SparseMatrix, ...)`. Uses an LDL^T
    /// factorisation (positive-semi-definite Cholesky variant) so that
    /// rank-deficient `X^T X` is handled gracefully — sklearn falls back
    /// to LSQR for sparse OLS, which is also robust to rank deficiency.
    ///
    /// Mirrors sklearn's `LinearRegression.fit` behaviour on sparse input.
    /// `sample_weight`, `positive`, `n_jobs` are documented parity gaps.
    template <int Options, typename StorageIndex>
    LinearRegression& fit(
        const Eigen::SparseMatrix<Scalar, Options, StorageIndex>& X,
        const Eigen::Ref<const VectorType>& y) {
        if (X.rows() == 0 || X.cols() == 0) {
            throw std::invalid_argument(
                "LinearRegression.fit: empty sparse matrix.");
        }
        if (X.rows() != y.size()) {
            throw std::invalid_argument(
                "LinearRegression.fit: X has " + std::to_string(X.rows()) +
                " rows but y has " + std::to_string(y.size()) + ".");
        }
        this->n_features_in_ = X.cols();

        using ColSparse =
            Eigen::SparseMatrix<Scalar, Eigen::ColMajor, StorageIndex>;
        const ColSparse Xs = X;
        const Eigen::Index n = Xs.rows();
        const Eigen::Index p = Xs.cols();

        RowVectorType x_mean = RowVectorType::Zero(p);
        if (fit_intercept_) {
            for (Eigen::Index j = 0; j < p; ++j) {
                Scalar s{0};
                for (typename ColSparse::InnerIterator it(Xs, j); it; ++it) {
                    s += it.value();
                }
                x_mean(j) = s / static_cast<Scalar>(n);
            }
        }
        const Scalar y_mean = fit_intercept_ ? y.mean() : Scalar{0};

        MatrixType XtX = MatrixType(Xs.transpose() * Xs);
        VectorType Xty = Xs.transpose() * y;
        if (fit_intercept_) {
            XtX.noalias() -= static_cast<Scalar>(n) *
                             (x_mean.transpose() * x_mean);
            Xty.noalias() -= static_cast<Scalar>(n) * y_mean *
                             x_mean.transpose();
        }

        Eigen::LDLT<MatrixType> ldlt(XtX);
        VectorType w = ldlt.solve(Xty);

        coef_ = w.transpose();
        // LDLT does not expose `.rank()`. We approximate the effective rank
        // via the count of non-tiny diagonal entries of D — the same heuristic
        // sklearn's sparse path applies in lieu of an explicit rank-revealing
        // factorisation.
        const auto vector_d = ldlt.vectorD();
        const Scalar tol = std::numeric_limits<Scalar>::epsilon() *
                           static_cast<Scalar>(vector_d.size()) *
                           vector_d.cwiseAbs().maxCoeff();
        Eigen::Index r = 0;
        for (Eigen::Index i = 0; i < vector_d.size(); ++i) {
            if (std::abs(vector_d(i)) > tol) ++r;
        }
        rank_ = r;

        if (fit_intercept_) {
            intercept_ = y_mean - x_mean.dot(w);
            x_offset_ = x_mean;
        } else {
            intercept_ = Scalar{0};
        }

        this->fitted_ = true;
        return *this;
    }

    // -- Multi-target regression (v1.1.0 §3.3) ------------------------------

    /// @brief Fit OLS with a multi-target response matrix.
    ///
    /// `Y` has shape (n_samples, n_targets). For each target column the OLS
    /// solution is computed via the same QR factorisation, exploiting the
    /// shared design matrix. The resulting coefficient matrix has shape
    /// (n_targets, n_features); the intercept is a vector of length
    /// `n_targets`.
    ///
    /// This overload is **additive** to the single-target API: the
    /// `coef()` / `intercept()` accessors continue to return the
    /// single-target row vector / scalar (extracted from the first column
    /// of `Y` when this overload was used). Multi-target callers use the
    /// new accessors `coef_matrix()` / `intercept_vector()` and the
    /// `predict_multi()` method.
    ///
    /// Mirrors sklearn's `LinearRegression.fit(X, Y)` behaviour for
    /// multi-output regression. `sample_weight`, `positive`, `n_jobs` are
    /// documented parity gaps.
    LinearRegression& fit_multi(const Eigen::Ref<const MatrixType>& X,
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

        if (fit_intercept_) {
            x_offset_ = X.colwise().mean();
            RowVectorType y_offset = Y.colwise().mean();

            MatrixType X_c = X.rowwise() - x_offset_;
            MatrixType Y_c = Y.rowwise() - y_offset;

            auto qr = X_c.colPivHouseholderQr();
            // qr.solve on a matrix RHS solves each column simultaneously.
            MatrixType W = qr.solve(Y_c);            // (n_features, n_targets)
            rank_ = qr.rank();

            // Coefficient matrix is (n_targets, n_features).
            coef_matrix_ = W.transpose();
            intercept_vector_ = y_offset.transpose() -
                                coef_matrix_ * x_offset_.transpose();
        } else {
            auto qr = X.colPivHouseholderQr();
            MatrixType W = qr.solve(Y);
            rank_ = qr.rank();
            coef_matrix_ = W.transpose();
            intercept_vector_ = VectorType::Zero(Y.cols());
        }

        // Mirror the first target into the single-target accessors so that
        // existing callers see consistent state when multi-target is used.
        coef_      = coef_matrix_.row(0);
        intercept_ = intercept_vector_(0);
        n_targets_ = static_cast<int>(Y.cols());

        this->fitted_ = true;
        return *this;
    }

    /// @brief Coefficient matrix of shape (n_targets, n_features).
    ///
    /// Populated by either `fit` (set to a 1-row matrix) or `fit_multi`.
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] const MatrixType& coef_matrix() const {
        this->check_is_fitted();
        if (coef_matrix_.size() == 0) {
            // Lazily synthesise the matrix view from the single-target state.
            coef_matrix_.resize(1, coef_.size());
            coef_matrix_.row(0) = coef_;
            intercept_vector_.resize(1);
            intercept_vector_(0) = intercept_;
            n_targets_ = 1;
        }
        return coef_matrix_;
    }

    /// @brief Intercept vector of length n_targets.
    /// @throws std::runtime_error if the model has not been fitted.
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

    /// @brief Number of targets seen during the most recent `fit` /
    ///   `fit_multi` call (1 for the single-target overload).
    [[nodiscard]] int n_targets() const {
        this->check_is_fitted(); return n_targets_;
    }

    /// @brief Predict multi-target outputs of shape (n_samples, n_targets).
    [[nodiscard]] MatrixType predict_multi(
        const Eigen::Ref<const MatrixType>& X) const {
        this->check_is_fitted();
        this->validate_feature_count(X);
        const MatrixType& W = coef_matrix();
        const VectorType& b = intercept_vector();
        // Y_pred = X * W^T + 1 * b^T
        MatrixType Y_pred = X * W.transpose();
        Y_pred.rowwise() += b.transpose();
        return Y_pred;
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
    ///
    /// @note **sklearn parity gap:** `sample_weight` parameter is
    ///   not yet supported.
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

private:
    bool fit_intercept_;

    RowVectorType coef_;
    Scalar intercept_ = Scalar{0};
    RowVectorType x_offset_;
    IndexType rank_ = 0;

    // Multi-target state (populated by fit_multi; mutable so that
    // coef_matrix() / intercept_vector() can lazily synthesise the
    // 1-target matrix views from the single-target state).
    mutable MatrixType  coef_matrix_;        // (n_targets, n_features)
    mutable VectorType  intercept_vector_;   // (n_targets,)
    mutable int         n_targets_ = 0;
};

/// @}

} // namespace Skigen

#endif // SKIGEN_LINEAR_MODEL_LINEAR_REGRESSION_H
