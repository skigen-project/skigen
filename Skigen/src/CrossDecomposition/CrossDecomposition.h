// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

#ifndef SKIGEN_CROSS_DECOMPOSITION_CROSS_DECOMPOSITION_H
#define SKIGEN_CROSS_DECOMPOSITION_CROSS_DECOMPOSITION_H

#include "../Core/Base.h"
#include "../Core/Validation.h"

#include <Eigen/Core>
#include <Eigen/QR>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace Skigen {

/// @defgroup Algo_CrossDecomposition Cross Decomposition
/// @ingroup CrossDecomposition
/// @brief Dense PLS and CCA cross-decomposition estimators.
/// @{

namespace cross_decomposition::detail {

template <typename Scalar>
using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

template <typename Scalar>
using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

template <typename Scalar>
using RowVector = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;

template <typename Scalar>
void check_target_matrix(const Matrix<Scalar>& target) {
    if (target.rows() == 0 || target.cols() == 0) {
        throw std::invalid_argument("CrossDecomposition: Y must be non-empty.");
    }
    if (!target.allFinite()) {
        throw std::invalid_argument("CrossDecomposition: Y must contain only finite values.");
    }
}

template <typename Scalar>
RowVector<Scalar> column_means(const Matrix<Scalar>& values) {
    return values.colwise().mean();
}

template <typename Scalar>
RowVector<Scalar> column_scales(const Matrix<Scalar>& centered, bool scale) {
    RowVector<Scalar> scales = RowVector<Scalar>::Ones(centered.cols());
    if (!scale) {
        return scales;
    }
    const Scalar denominator = static_cast<Scalar>(centered.rows() - 1);
    for (Eigen::Index col = 0; col < centered.cols(); ++col) {
        const Scalar variance = centered.col(col).squaredNorm() / denominator;
        const Scalar stddev = std::sqrt(std::max(variance, Scalar{0}));
        scales(col) = stddev > Scalar{0} ? stddev : Scalar{1};
    }
    return scales;
}

template <typename Scalar>
Matrix<Scalar> center_scale(const Matrix<Scalar>& values,
                            const RowVector<Scalar>& means,
                            const RowVector<Scalar>& scales) {
    Matrix<Scalar> normalized = values.rowwise() - means;
    normalized.array().rowwise() /= scales.array();
    return normalized;
}

template <typename Scalar>
Vector<Scalar> first_nonzero_column(const Matrix<Scalar>& values) {
    for (Eigen::Index col = 0; col < values.cols(); ++col) {
        if (values.col(col).squaredNorm() > std::numeric_limits<Scalar>::epsilon()) {
            return values.col(col);
        }
    }
    return Vector<Scalar>::Zero(values.rows());
}

template <typename Scalar>
void normalize_or_throw(Vector<Scalar>& values, const char* message) {
    const Scalar norm = values.norm();
    if (!(norm > std::numeric_limits<Scalar>::epsilon())) {
        throw std::runtime_error(message);
    }
    values /= norm;
}

template <typename Scalar>
void orient_component(Vector<Scalar>& input_weights,
                      Vector<Scalar>& input_scores,
                      Vector<Scalar>& target_weights,
                      Vector<Scalar>& target_scores) {
    Eigen::Index max_index = 0;
    input_weights.cwiseAbs().maxCoeff(&max_index);
    if (input_weights(max_index) < Scalar{0}) {
        input_weights = -input_weights;
        input_scores = -input_scores;
        target_weights = -target_weights;
        target_scores = -target_scores;
    }
}

template <typename Scalar>
Matrix<Scalar> right_inverse(const Matrix<Scalar>& factor) {
    const Matrix<Scalar> identity = Matrix<Scalar>::Identity(factor.rows(), factor.cols());
    return factor.completeOrthogonalDecomposition().solve(identity);
}

template <typename Scalar>
Scalar r2_score(const Matrix<Scalar>& truth, const Matrix<Scalar>& predictions) {
    const RowVector<Scalar> means = truth.colwise().mean();
    const Scalar ss_res = (truth - predictions).squaredNorm();
    const Scalar ss_tot = (truth.rowwise() - means).squaredNorm();
    return ss_tot > Scalar{0} ? Scalar{1} - ss_res / ss_tot : Scalar{0};
}

}  // namespace cross_decomposition::detail

/// @brief Partial least squares regression for dense multi-output targets.
///
/// Fits latent components with the NIPALS algorithm and exposes sklearn-style
/// weights, loadings, scores, rotations, coefficients, and intercepts.
///
/// Mirrors the dense core of `sklearn.cross_decomposition.PLSRegression`:
/// https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html.
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `n_components` | `int` | `2` | Number of latent components. |
/// | `scale` | `bool` | `true` | Center and variance-scale `X` and `Y`. |
/// | `max_iter` | `int` | `500` | Maximum NIPALS iterations per component. |
/// | `tol` | `Scalar` | `1e-6` | Squared-score convergence tolerance. |
///
/// ### Examples
///
/// @snippet cross_decomposition.cpp example_pls_regression
template <typename Scalar = double>
class PLSRegression : public Estimator<PLSRegression<Scalar>, Scalar> {
public:
    using Base = Estimator<PLSRegression<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;

    explicit PLSRegression(int n_components = 2,
                           bool scale = true,
                           int max_iter = 500,
                           Scalar tol = Scalar{1e-6})
        : n_components_(n_components), scale_(scale), max_iter_(max_iter), tol_(tol) {}

    [[nodiscard]] int n_components() const noexcept { return n_components_; }
    [[nodiscard]] bool scale() const noexcept { return scale_; }
    [[nodiscard]] int max_iter() const noexcept { return max_iter_; }
    [[nodiscard]] Scalar tol() const noexcept { return tol_; }

    [[nodiscard]] const MatrixType& x_weights() const { this->check_is_fitted(); return x_weights_; }
    [[nodiscard]] const MatrixType& y_weights() const { this->check_is_fitted(); return y_weights_; }
    [[nodiscard]] const MatrixType& x_loadings() const { this->check_is_fitted(); return x_loadings_; }
    [[nodiscard]] const MatrixType& y_loadings() const { this->check_is_fitted(); return y_loadings_; }
    [[nodiscard]] const MatrixType& x_scores() const { this->check_is_fitted(); return x_scores_; }
    [[nodiscard]] const MatrixType& y_scores() const { this->check_is_fitted(); return y_scores_; }
    [[nodiscard]] const MatrixType& x_rotations() const { this->check_is_fitted(); return x_rotations_; }
    [[nodiscard]] const MatrixType& y_rotations() const { this->check_is_fitted(); return y_rotations_; }
    [[nodiscard]] const MatrixType& coef() const { this->check_is_fitted(); return coef_; }
    [[nodiscard]] const RowVectorType& intercept() const { this->check_is_fitted(); return intercept_; }
    [[nodiscard]] const Eigen::VectorXi& n_iter() const { this->check_is_fitted(); return n_iter_; }

    SKIGEN_PARAMS(
        (n_components, n_components_, int),
        (scale, scale_, bool),
        (max_iter, max_iter_, int),
        (tol, tol_, double))

    PLSRegression& fit(const Eigen::Ref<const MatrixType>& input,
                       const Eigen::Ref<const MatrixType>& target) {
        fit_impl(input, target);
        return *this;
    }

    [[nodiscard]] MatrixType transform(const Eigen::Ref<const MatrixType>& input) const {
        this->check_is_fitted();
        this->validate_feature_count(input);
        return transform_impl(input);
    }

    [[nodiscard]] std::pair<MatrixType, MatrixType> transform(
        const Eigen::Ref<const MatrixType>& input,
        const Eigen::Ref<const MatrixType>& target) const {
        this->check_is_fitted();
        this->validate_feature_count(input);
        if (target.cols() != y_mean_.cols()) {
            throw std::invalid_argument("PLSRegression.transform: Y feature count mismatch.");
        }
        const MatrixType target_scaled = cross_decomposition::detail::center_scale(MatrixType(target), y_mean_, y_std_);
        return {transform_impl(input), target_scaled * y_rotations_};
    }

    [[nodiscard]] MatrixType fit_transform(const Eigen::Ref<const MatrixType>& input,
                                           const Eigen::Ref<const MatrixType>& target) {
        fit(input, target);
        return transform_impl(input);
    }

    [[nodiscard]] MatrixType predict(const Eigen::Ref<const MatrixType>& input) const {
        this->check_is_fitted();
        this->validate_feature_count(input);
        return (input * coef_).rowwise() + intercept_;
    }

    [[nodiscard]] Scalar score(const Eigen::Ref<const MatrixType>& input,
                               const Eigen::Ref<const MatrixType>& target) const {
        this->check_is_fitted();
        this->validate_feature_count(input);
        return cross_decomposition::detail::r2_score(MatrixType(target), predict(input));
    }

private:
    PLSRegression& fit_impl(const Eigen::Ref<const MatrixType>& input,
                            const Eigen::Ref<const MatrixType>& target) {
        validate_fit_input(input, target);
        this->n_features_in_ = input.cols();

        x_mean_ = cross_decomposition::detail::column_means(MatrixType(input));
        y_mean_ = cross_decomposition::detail::column_means(MatrixType(target));
        MatrixType input_residual = MatrixType(input).rowwise() - x_mean_;
        MatrixType target_residual = MatrixType(target).rowwise() - y_mean_;
        x_std_ = cross_decomposition::detail::column_scales(input_residual, scale_);
        y_std_ = cross_decomposition::detail::column_scales(target_residual, scale_);
        input_residual.array().rowwise() /= x_std_.array();
        target_residual.array().rowwise() /= y_std_.array();

        allocate_state(input.rows(), input.cols(), target.cols());
        for (int component = 0; component < n_components_; ++component) {
            fit_component(component, input_residual, target_residual);
        }
        finalize_coefficients();
        this->fitted_ = true;
        return *this;
    }

    void validate_fit_input(const Eigen::Ref<const MatrixType>& input,
                            const Eigen::Ref<const MatrixType>& target) const {
        internal::check_non_empty(input);
        internal::check_consistent_length(input, target);
        if (!input.allFinite()) {
            throw std::invalid_argument("PLSRegression: X must contain only finite values.");
        }
        cross_decomposition::detail::check_target_matrix(MatrixType(target));
        if (input.rows() < 2) {
            throw std::invalid_argument("PLSRegression: at least two samples are required.");
        }
        if (n_components_ <= 0) {
            throw std::invalid_argument("PLSRegression: n_components must be positive.");
        }
        const IndexType max_components = std::min(input.rows() - 1, input.cols());
        if (n_components_ > max_components) {
            throw std::invalid_argument("PLSRegression: n_components is too large for X.");
        }
        if (max_iter_ <= 0) {
            throw std::invalid_argument("PLSRegression: max_iter must be positive.");
        }
        if (tol_ < Scalar{0}) {
            throw std::invalid_argument("PLSRegression: tol must be non-negative.");
        }
    }

    void allocate_state(IndexType rows, IndexType input_cols, IndexType target_cols) {
        x_weights_.setZero(input_cols, n_components_);
        y_weights_.setZero(target_cols, n_components_);
        x_loadings_.setZero(input_cols, n_components_);
        y_loadings_.setZero(target_cols, n_components_);
        x_scores_.setZero(rows, n_components_);
        y_scores_.setZero(rows, n_components_);
        x_rotations_.setZero(input_cols, n_components_);
        y_rotations_.setZero(target_cols, n_components_);
        coef_.setZero(input_cols, target_cols);
        intercept_.setZero(target_cols);
        n_iter_.setZero(n_components_);
    }

    void fit_component(int component, MatrixType& input_residual, MatrixType& target_residual) {
        using cross_decomposition::detail::normalize_or_throw;
        VectorType target_score = cross_decomposition::detail::first_nonzero_column(target_residual);
        if (target_score.squaredNorm() <= std::numeric_limits<Scalar>::epsilon()) {
            throw std::runtime_error("PLSRegression: residual Y became rank deficient.");
        }

        VectorType input_weight(input_residual.cols());
        VectorType target_weight(target_residual.cols());
        VectorType input_score(input_residual.rows());
        for (int iteration = 0; iteration < max_iter_; ++iteration) {
            const VectorType previous_target_score = target_score;
            input_weight = input_residual.transpose() * target_score / target_score.squaredNorm();
            normalize_or_throw(input_weight, "PLSRegression: X residual is rank deficient.");
            input_score = input_residual * input_weight;
            if (input_score.squaredNorm() <= std::numeric_limits<Scalar>::epsilon()) {
                throw std::runtime_error("PLSRegression: latent X score is degenerate.");
            }
            target_weight = target_residual.transpose() * input_score / input_score.squaredNorm();
            normalize_or_throw(target_weight, "PLSRegression: Y residual is rank deficient.");
            target_score = target_residual * target_weight;
            n_iter_(component) = iteration + 1;
            if ((target_score - previous_target_score).squaredNorm() <= tol_ * tol_) {
                break;
            }
        }

        cross_decomposition::detail::orient_component(input_weight, input_score, target_weight, target_score);
        const Scalar score_norm = input_score.squaredNorm();
        const VectorType input_loading = input_residual.transpose() * input_score / score_norm;
        const VectorType target_loading = target_residual.transpose() * input_score / score_norm;

        input_residual -= input_score * input_loading.transpose();
        target_residual -= input_score * target_loading.transpose();

        x_weights_.col(component) = input_weight;
        y_weights_.col(component) = target_weight;
        x_loadings_.col(component) = input_loading;
        y_loadings_.col(component) = target_loading;
        x_scores_.col(component) = input_score;
        y_scores_.col(component) = target_score;
    }

    void finalize_coefficients() {
        const MatrixType x_cross = x_loadings_.transpose() * x_weights_;
        const MatrixType y_cross = y_loadings_.transpose() * y_weights_;
        x_rotations_ = x_weights_ * cross_decomposition::detail::right_inverse(x_cross);
        y_rotations_ = y_weights_ * cross_decomposition::detail::right_inverse(y_cross);
        MatrixType scaled_coef = x_rotations_ * y_loadings_.transpose();
        coef_ = x_std_.cwiseInverse().asDiagonal() * scaled_coef * y_std_.asDiagonal();
        intercept_ = y_mean_ - x_mean_ * coef_;
    }

    [[nodiscard]] MatrixType transform_impl(const Eigen::Ref<const MatrixType>& input) const {
        const MatrixType input_scaled = cross_decomposition::detail::center_scale(MatrixType(input), x_mean_, x_std_);
        return input_scaled * x_rotations_;
    }

    int n_components_;
    bool scale_;
    int max_iter_;
    Scalar tol_;
    RowVectorType x_mean_;
    RowVectorType y_mean_;
    RowVectorType x_std_;
    RowVectorType y_std_;
    MatrixType x_weights_;
    MatrixType y_weights_;
    MatrixType x_loadings_;
    MatrixType y_loadings_;
    MatrixType x_scores_;
    MatrixType y_scores_;
    MatrixType x_rotations_;
    MatrixType y_rotations_;
    MatrixType coef_;
    RowVectorType intercept_;
    Eigen::VectorXi n_iter_;
};

/// @brief Canonical correlation analysis for dense paired datasets.
///
/// Uses the NIPALS mode-B update to learn paired latent projections of `X` and
/// `Y` with maximal linear correlation.
///
/// Mirrors the dense core of `sklearn.cross_decomposition.CCA`:
/// https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.CCA.html.
///
/// ### Examples
///
/// @snippet cross_decomposition.cpp example_cca
template <typename Scalar = double>
class CCA : public Estimator<CCA<Scalar>, Scalar> {
public:
    using Base = Estimator<CCA<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;

    explicit CCA(int n_components = 2,
                 bool scale = true,
                 int max_iter = 500,
                 Scalar tol = Scalar{1e-6})
        : n_components_(n_components), scale_(scale), max_iter_(max_iter), tol_(tol) {}

    [[nodiscard]] int n_components() const noexcept { return n_components_; }
    [[nodiscard]] bool scale() const noexcept { return scale_; }
    [[nodiscard]] int max_iter() const noexcept { return max_iter_; }
    [[nodiscard]] Scalar tol() const noexcept { return tol_; }

    [[nodiscard]] const MatrixType& x_weights() const { this->check_is_fitted(); return x_weights_; }
    [[nodiscard]] const MatrixType& y_weights() const { this->check_is_fitted(); return y_weights_; }
    [[nodiscard]] const MatrixType& x_loadings() const { this->check_is_fitted(); return x_loadings_; }
    [[nodiscard]] const MatrixType& y_loadings() const { this->check_is_fitted(); return y_loadings_; }
    [[nodiscard]] const MatrixType& x_scores() const { this->check_is_fitted(); return x_scores_; }
    [[nodiscard]] const MatrixType& y_scores() const { this->check_is_fitted(); return y_scores_; }
    [[nodiscard]] const MatrixType& x_rotations() const { this->check_is_fitted(); return x_rotations_; }
    [[nodiscard]] const MatrixType& y_rotations() const { this->check_is_fitted(); return y_rotations_; }
    [[nodiscard]] const MatrixType& coef() const { this->check_is_fitted(); return coef_; }
    [[nodiscard]] const RowVectorType& intercept() const { this->check_is_fitted(); return intercept_; }
    [[nodiscard]] const Eigen::VectorXi& n_iter() const { this->check_is_fitted(); return n_iter_; }

    SKIGEN_PARAMS(
        (n_components, n_components_, int),
        (scale, scale_, bool),
        (max_iter, max_iter_, int),
        (tol, tol_, double))

    CCA& fit(const Eigen::Ref<const MatrixType>& input,
             const Eigen::Ref<const MatrixType>& target) {
        fit_impl(input, target);
        return *this;
    }

    [[nodiscard]] std::pair<MatrixType, MatrixType> transform(
        const Eigen::Ref<const MatrixType>& input,
        const Eigen::Ref<const MatrixType>& target) const {
        this->check_is_fitted();
        this->validate_feature_count(input);
        if (target.cols() != y_mean_.cols()) {
            throw std::invalid_argument("CCA.transform: Y feature count mismatch.");
        }
        const MatrixType input_scaled = cross_decomposition::detail::center_scale(MatrixType(input), x_mean_, x_std_);
        const MatrixType target_scaled = cross_decomposition::detail::center_scale(MatrixType(target), y_mean_, y_std_);
        return {input_scaled * x_rotations_, target_scaled * y_rotations_};
    }

    [[nodiscard]] MatrixType transform(const Eigen::Ref<const MatrixType>& input) const {
        this->check_is_fitted();
        this->validate_feature_count(input);
        const MatrixType input_scaled = cross_decomposition::detail::center_scale(MatrixType(input), x_mean_, x_std_);
        return input_scaled * x_rotations_;
    }

    [[nodiscard]] MatrixType predict(const Eigen::Ref<const MatrixType>& input) const {
        this->check_is_fitted();
        this->validate_feature_count(input);
        return (input * coef_).rowwise() + intercept_;
    }

    [[nodiscard]] Scalar score(const Eigen::Ref<const MatrixType>& input,
                               const Eigen::Ref<const MatrixType>& target) const {
        this->check_is_fitted();
        this->validate_feature_count(input);
        return cross_decomposition::detail::r2_score(MatrixType(target), predict(input));
    }

private:
    CCA& fit_impl(const Eigen::Ref<const MatrixType>& input,
                  const Eigen::Ref<const MatrixType>& target) {
        validate_fit_input(input, target);
        this->n_features_in_ = input.cols();

        x_mean_ = cross_decomposition::detail::column_means(MatrixType(input));
        y_mean_ = cross_decomposition::detail::column_means(MatrixType(target));
        MatrixType input_residual = MatrixType(input).rowwise() - x_mean_;
        MatrixType target_residual = MatrixType(target).rowwise() - y_mean_;
        x_std_ = cross_decomposition::detail::column_scales(input_residual, scale_);
        y_std_ = cross_decomposition::detail::column_scales(target_residual, scale_);
        input_residual.array().rowwise() /= x_std_.array();
        target_residual.array().rowwise() /= y_std_.array();

        allocate_state(input.rows(), input.cols(), target.cols());
        for (int component = 0; component < n_components_; ++component) {
            fit_component(component, input_residual, target_residual);
        }
        finalize_coefficients();
        this->fitted_ = true;
        return *this;
    }

    void validate_fit_input(const Eigen::Ref<const MatrixType>& input,
                            const Eigen::Ref<const MatrixType>& target) const {
        internal::check_non_empty(input);
        internal::check_consistent_length(input, target);
        if (!input.allFinite()) {
            throw std::invalid_argument("CCA: X must contain only finite values.");
        }
        cross_decomposition::detail::check_target_matrix(MatrixType(target));
        if (input.rows() < 2) {
            throw std::invalid_argument("CCA: at least two samples are required.");
        }
        if (n_components_ <= 0) {
            throw std::invalid_argument("CCA: n_components must be positive.");
        }
        const IndexType max_components = std::min(input.cols(), target.cols());
        if (n_components_ > max_components) {
            throw std::invalid_argument("CCA: n_components is too large for X and Y.");
        }
        if (max_iter_ <= 0) {
            throw std::invalid_argument("CCA: max_iter must be positive.");
        }
        if (tol_ < Scalar{0}) {
            throw std::invalid_argument("CCA: tol must be non-negative.");
        }
    }

    void allocate_state(IndexType rows, IndexType input_cols, IndexType target_cols) {
        x_weights_.setZero(input_cols, n_components_);
        y_weights_.setZero(target_cols, n_components_);
        x_loadings_.setZero(input_cols, n_components_);
        y_loadings_.setZero(target_cols, n_components_);
        x_scores_.setZero(rows, n_components_);
        y_scores_.setZero(rows, n_components_);
        x_rotations_.setZero(input_cols, n_components_);
        y_rotations_.setZero(target_cols, n_components_);
        coef_.setZero(input_cols, target_cols);
        intercept_.setZero(target_cols);
        n_iter_.setZero(n_components_);
    }

    void fit_component(int component, MatrixType& input_residual, MatrixType& target_residual) {
        using cross_decomposition::detail::normalize_or_throw;
        VectorType target_score = cross_decomposition::detail::first_nonzero_column(target_residual);
        if (target_score.squaredNorm() <= std::numeric_limits<Scalar>::epsilon()) {
            throw std::runtime_error("CCA: residual Y became rank deficient.");
        }

        VectorType input_weight(input_residual.cols());
        VectorType target_weight(target_residual.cols());
        VectorType input_score(input_residual.rows());
        for (int iteration = 0; iteration < max_iter_; ++iteration) {
            const VectorType previous_target_score = target_score;
            input_weight = input_residual.completeOrthogonalDecomposition().solve(target_score);
            normalize_or_throw(input_weight, "CCA: X residual is rank deficient.");
            input_score = input_residual * input_weight;
            if (input_score.squaredNorm() <= std::numeric_limits<Scalar>::epsilon()) {
                throw std::runtime_error("CCA: latent X score is degenerate.");
            }
            target_weight = target_residual.completeOrthogonalDecomposition().solve(input_score);
            normalize_or_throw(target_weight, "CCA: Y residual is rank deficient.");
            target_score = target_residual * target_weight;
            n_iter_(component) = iteration + 1;
            if ((target_score - previous_target_score).squaredNorm() <= tol_ * tol_) {
                break;
            }
        }

        cross_decomposition::detail::orient_component(input_weight, input_score, target_weight, target_score);
        const Scalar input_score_norm = input_score.squaredNorm();
        const Scalar target_score_norm = target_score.squaredNorm();
        const VectorType input_loading = input_residual.transpose() * input_score / input_score_norm;
        const VectorType target_loading = target_residual.transpose() * target_score / target_score_norm;

        input_residual -= input_score * input_loading.transpose();
        target_residual -= target_score * target_loading.transpose();

        x_weights_.col(component) = input_weight;
        y_weights_.col(component) = target_weight;
        x_loadings_.col(component) = input_loading;
        y_loadings_.col(component) = target_loading;
        x_scores_.col(component) = input_score;
        y_scores_.col(component) = target_score;
    }

    void finalize_coefficients() {
        const MatrixType x_cross = x_loadings_.transpose() * x_weights_;
        const MatrixType y_cross = y_loadings_.transpose() * y_weights_;
        x_rotations_ = x_weights_ * cross_decomposition::detail::right_inverse(x_cross);
        y_rotations_ = y_weights_ * cross_decomposition::detail::right_inverse(y_cross);
        MatrixType scaled_coef = x_rotations_ * y_loadings_.transpose();
        coef_ = x_std_.cwiseInverse().asDiagonal() * scaled_coef * y_std_.asDiagonal();
        intercept_ = y_mean_ - x_mean_ * coef_;
    }

    int n_components_;
    bool scale_;
    int max_iter_;
    Scalar tol_;
    RowVectorType x_mean_;
    RowVectorType y_mean_;
    RowVectorType x_std_;
    RowVectorType y_std_;
    MatrixType x_weights_;
    MatrixType y_weights_;
    MatrixType x_loadings_;
    MatrixType y_loadings_;
    MatrixType x_scores_;
    MatrixType y_scores_;
    MatrixType x_rotations_;
    MatrixType y_rotations_;
    MatrixType coef_;
    RowVectorType intercept_;
    Eigen::VectorXi n_iter_;
};

/// @}

}  // namespace Skigen

#endif  // SKIGEN_CROSS_DECOMPOSITION_CROSS_DECOMPOSITION_H
