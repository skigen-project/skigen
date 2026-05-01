// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_LINEAR_MODEL_ELASTIC_NET_H
#define SKIGEN_LINEAR_MODEL_ELASTIC_NET_H

#include "../Core/Base.h"
#include "../Core/Validation.h"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>

namespace Skigen {

/// ElasticNet — Linear regression with combined L1 and L2 regularization.
/// Mirrors sklearn.linear_model.ElasticNet.
///
/// Objective: (1/2n) ||y - Xw||² + alpha * l1_ratio * ||w||₁
///                                 + alpha * (1-l1_ratio)/2 * ||w||²
template <typename Scalar = double>
class ElasticNet
    : public Predictor<ElasticNet<Scalar>, Scalar> {
public:
    using Base = Predictor<ElasticNet<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;

    explicit ElasticNet(Scalar alpha = Scalar{1}, Scalar l1_ratio = Scalar{0.5},
                        bool fit_intercept = true,
                        int max_iter = 1000, Scalar tol = Scalar{1e-4})
        : alpha_(alpha), l1_ratio_(l1_ratio), fit_intercept_(fit_intercept),
          max_iter_(max_iter), tol_(tol) {}

    [[nodiscard]] const RowVectorType& coef() const {
        this->check_is_fitted(); return coef_;
    }
    [[nodiscard]] Scalar intercept() const {
        this->check_is_fitted(); return intercept_;
    }

    ElasticNet& fit_impl(const Eigen::Ref<const MatrixType>& X,
                         const Eigen::Ref<const VectorType>& y) {
        internal::check_non_empty(X);
        this->n_features_in_ = X.cols();
        const Eigen::Index n = X.rows();
        const Eigen::Index p = X.cols();

        RowVectorType X_mean = RowVectorType::Zero(p);
        Scalar y_mean{0};

        if (fit_intercept_) {
            X_mean = X.colwise().mean();
            y_mean = y.mean();
        }

        MatrixType X_c;
        VectorType y_c;

        if (fit_intercept_) {
            X_c = (X.rowwise() - X_mean).eval();
            y_c = (y.array() - y_mean).matrix().eval();
        } else {
            X_c = X;
            y_c = y;
        }

        VectorType col_norms_sq(p);
        for (Eigen::Index j = 0; j < p; ++j) {
            col_norms_sq(j) = X_c.col(j).squaredNorm();
        }

        Scalar l1_penalty = static_cast<Scalar>(n) * alpha_ * l1_ratio_;
        Scalar l2_penalty = static_cast<Scalar>(n) * alpha_ * (Scalar{1} - l1_ratio_);

        coef_ = RowVectorType::Zero(p);
        VectorType residual = y_c;

        for (int iter = 0; iter < max_iter_; ++iter) {
            Scalar max_change{0};

            for (Eigen::Index j = 0; j < p; ++j) {
                Scalar denom = col_norms_sq(j) + l2_penalty;
                if (denom < std::numeric_limits<Scalar>::epsilon()) continue;

                Scalar old_w = coef_(j);
                Scalar rho = X_c.col(j).dot(residual) + col_norms_sq(j) * old_w;

                coef_(j) = soft_threshold(rho, l1_penalty) / denom;

                Scalar delta = coef_(j) - old_w;
                if (delta != Scalar{0}) {
                    residual -= delta * X_c.col(j);
                }

                max_change = std::max(max_change, std::abs(delta));
            }

            if (max_change < tol_) break;
        }

        intercept_ = fit_intercept_
            ? y_mean - (X_mean.array() * coef_.array()).sum()
            : Scalar{0};

        this->fitted_ = true;
        return *this;
    }

    [[nodiscard]] VectorType predict_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        return (X * coef_.transpose()).array() + intercept_;
    }

    [[nodiscard]] Scalar score_impl(const Eigen::Ref<const MatrixType>& X,
                                    const Eigen::Ref<const VectorType>& y) const {
        VectorType y_pred = predict_impl(X);
        Scalar ss_res = (y - y_pred).squaredNorm();
        Scalar ss_tot = (y.array() - y.mean()).matrix().squaredNorm();
        if (ss_tot == Scalar{0}) return Scalar{0};
        return Scalar{1} - ss_res / ss_tot;
    }

private:
    Scalar alpha_;
    Scalar l1_ratio_;
    bool fit_intercept_;
    int max_iter_;
    Scalar tol_;

    RowVectorType coef_;
    Scalar intercept_{0};

    static Scalar soft_threshold(Scalar x, Scalar lambda) {
        if (x > lambda) return x - lambda;
        if (x < -lambda) return x + lambda;
        return Scalar{0};
    }
};

} // namespace Skigen

#endif // SKIGEN_LINEAR_MODEL_ELASTIC_NET_H
