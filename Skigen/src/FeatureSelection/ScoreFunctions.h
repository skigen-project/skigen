// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_FEATURE_SELECTION_SCORE_FUNCTIONS_H
#define SKIGEN_FEATURE_SELECTION_SCORE_FUNCTIONS_H

#include "../Core/Validation.h"

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace Skigen {
namespace feature_selection {

namespace detail {

// ---------------------------------------------------------------------------
// Special function approximations.
//
// We need:
//   - Regularized lower incomplete gamma P(a, x) for chi-squared CDF
//   - Regularized incomplete beta I_x(a, b) for F-distribution CDF
//
// Implementations adapted from "Numerical Recipes" style routines,
// rewritten for templates with adequate precision (better than 1e-10).
// ---------------------------------------------------------------------------

template <typename Scalar>
Scalar log_gamma(Scalar x) {
    // Lanczos approximation (g = 7, n = 9)
    static const double cof[8] = {
         676.5203681218851,    -1259.1392167224028,
         771.32342877765313,   -176.61502916214059,
         12.507343278686905,   -0.13857109526572012,
         9.9843695780195716e-6, 1.5056327351493116e-7
    };
    if (x < Scalar{0.5}) {
        // Reflection: log Γ(x) = log(π/sin(πx)) − log Γ(1−x)
        const double pi = 3.141592653589793238462643383279502884;
        return static_cast<Scalar>(std::log(pi /
                std::sin(pi * static_cast<double>(x)))) -
               log_gamma(Scalar{1} - x);
    }
    double xd = static_cast<double>(x) - 1.0;
    double a = 0.99999999999980993;
    double t = xd + 7.5;
    for (int i = 0; i < 8; ++i) {
        a += cof[i] / (xd + static_cast<double>(i + 1));
    }
    const double half_log_2pi = 0.91893853320467274178;  // 0.5 * log(2π)
    return static_cast<Scalar>(half_log_2pi + (xd + 0.5) * std::log(t) - t +
                               std::log(a));
}

// Regularized lower incomplete gamma P(a, x) via series expansion
// Valid when x < a + 1.
template <typename Scalar>
Scalar gamma_p_series(Scalar a, Scalar x) {
    if (x <= Scalar{0}) return Scalar{0};
    const int max_iter = 200;
    const double eps = 3e-15;
    double ap = static_cast<double>(a);
    double sum = 1.0 / ap;
    double del = sum;
    for (int n = 1; n < max_iter; ++n) {
        ap += 1.0;
        del *= static_cast<double>(x) / ap;
        sum += del;
        if (std::fabs(del) < std::fabs(sum) * eps) break;
    }
    double res = sum * std::exp(-static_cast<double>(x) +
                                static_cast<double>(a) *
                                std::log(static_cast<double>(x)) -
                                static_cast<double>(log_gamma(a)));
    return static_cast<Scalar>(res);
}

// Regularized upper incomplete gamma Q(a, x) via continued fraction
// Valid when x >= a + 1.
template <typename Scalar>
Scalar gamma_q_cf(Scalar a, Scalar x) {
    const int max_iter = 200;
    const double eps = 3e-15;
    const double fpmin = 1e-300;
    double ad = static_cast<double>(a);
    double xd = static_cast<double>(x);
    double b = xd + 1.0 - ad;
    double c = 1.0 / fpmin;
    double d = 1.0 / b;
    double h = d;
    for (int i = 1; i < max_iter; ++i) {
        double an = -static_cast<double>(i) * (static_cast<double>(i) - ad);
        b += 2.0;
        d = an * d + b;
        if (std::fabs(d) < fpmin) d = fpmin;
        c = b + an / c;
        if (std::fabs(c) < fpmin) c = fpmin;
        d = 1.0 / d;
        double del = d * c;
        h *= del;
        if (std::fabs(del - 1.0) < eps) break;
    }
    double res = std::exp(-xd + ad * std::log(xd) -
                          static_cast<double>(log_gamma(a))) * h;
    return static_cast<Scalar>(res);
}

template <typename Scalar>
Scalar gamma_p(Scalar a, Scalar x) {
    if (x < Scalar{0} || a <= Scalar{0}) return Scalar{0};
    if (x == Scalar{0}) return Scalar{0};
    if (static_cast<double>(x) < static_cast<double>(a) + 1.0) {
        return gamma_p_series(a, x);
    } else {
        return Scalar{1} - gamma_q_cf(a, x);
    }
}

template <typename Scalar>
Scalar gamma_q(Scalar a, Scalar x) {
    return Scalar{1} - gamma_p(a, x);
}

// Regularized incomplete beta function I_x(a, b) via continued fraction
template <typename Scalar>
Scalar beta_cf(Scalar a, Scalar b, Scalar x) {
    const int max_iter = 400;
    const double eps = 3e-15;
    const double fpmin = 1e-300;
    double ad = static_cast<double>(a);
    double bd = static_cast<double>(b);
    double xd = static_cast<double>(x);
    double qab = ad + bd;
    double qap = ad + 1.0;
    double qam = ad - 1.0;
    double c = 1.0;
    double d = 1.0 - qab * xd / qap;
    if (std::fabs(d) < fpmin) d = fpmin;
    d = 1.0 / d;
    double h = d;
    for (int m = 1; m <= max_iter; ++m) {
        int m2 = 2 * m;
        double aa = static_cast<double>(m) * (bd - static_cast<double>(m)) *
                    xd /
                    ((qam + static_cast<double>(m2)) *
                     (ad + static_cast<double>(m2)));
        d = 1.0 + aa * d;
        if (std::fabs(d) < fpmin) d = fpmin;
        c = 1.0 + aa / c;
        if (std::fabs(c) < fpmin) c = fpmin;
        d = 1.0 / d;
        h *= d * c;
        aa = -(ad + static_cast<double>(m)) *
             (qab + static_cast<double>(m)) * xd /
             ((ad + static_cast<double>(m2)) *
              (qap + static_cast<double>(m2)));
        d = 1.0 + aa * d;
        if (std::fabs(d) < fpmin) d = fpmin;
        c = 1.0 + aa / c;
        if (std::fabs(c) < fpmin) c = fpmin;
        d = 1.0 / d;
        double del = d * c;
        h *= del;
        if (std::fabs(del - 1.0) < eps) break;
    }
    return static_cast<Scalar>(h);
}

template <typename Scalar>
Scalar beta_inc_reg(Scalar a, Scalar b, Scalar x) {
    if (x <= Scalar{0}) return Scalar{0};
    if (x >= Scalar{1}) return Scalar{1};
    double ad = static_cast<double>(a);
    double bd = static_cast<double>(b);
    double xd = static_cast<double>(x);
    double bt = std::exp(static_cast<double>(log_gamma(a + b)) -
                         static_cast<double>(log_gamma(a)) -
                         static_cast<double>(log_gamma(b)) +
                         ad * std::log(xd) +
                         bd * std::log(1.0 - xd));
    if (xd < (ad + 1.0) / (ad + bd + 2.0)) {
        return static_cast<Scalar>(bt) * beta_cf(a, b, x) / a;
    } else {
        return Scalar{1} -
               static_cast<Scalar>(bt) * beta_cf(b, a, Scalar{1} - x) / b;
    }
}

// Survival functions (1 - CDF)
template <typename Scalar>
Scalar chi2_sf(Scalar x, Scalar dof) {
    // chi2 SF = Q(dof/2, x/2)
    if (x <= Scalar{0}) return Scalar{1};
    return gamma_q(dof / Scalar{2}, x / Scalar{2});
}

template <typename Scalar>
Scalar f_sf(Scalar f, Scalar dfn, Scalar dfd) {
    // F-distribution SF: 1 - I_{dfn*f/(dfn*f + dfd)}(dfn/2, dfd/2)
    //                  = I_{dfd/(dfd + dfn*f)}(dfd/2, dfn/2)
    if (f <= Scalar{0}) return Scalar{1};
    Scalar x = dfd / (dfd + dfn * f);
    return beta_inc_reg(dfd / Scalar{2}, dfn / Scalar{2}, x);
}

}  // namespace detail

// ---------------------------------------------------------------------------
// f_classif — ANOVA F-value between feature and target (classification).
// ---------------------------------------------------------------------------
/// @brief Compute the ANOVA F-value for the provided sample.
///
/// Mirrors
/// [sklearn.feature_selection.f_classif](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html).
///
/// @param X Design matrix of shape (n_samples, n_features).
/// @param y Integer class labels of shape (n_samples,).
/// @return Pair `(F, p)` where `F` is the F-statistic per feature and `p`
///   is the corresponding p-value, both as `RowVectorType` of length
///   n_features.
template <typename Scalar>
std::pair<Eigen::Matrix<Scalar, 1, Eigen::Dynamic>,
          Eigen::Matrix<Scalar, 1, Eigen::Dynamic>>
f_classif(const Eigen::Ref<const Eigen::Matrix<
              Scalar, Eigen::Dynamic, Eigen::Dynamic>>& X,
          const Eigen::Ref<const Eigen::VectorXi>& y) {
    using RowVec = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;
    internal::check_non_empty(X);
    internal::check_consistent_length(X, y);

    const Eigen::Index n = X.rows();
    const Eigen::Index p = X.cols();

    // Group samples by class label.
    std::map<int, std::vector<Eigen::Index>> groups;
    for (Eigen::Index i = 0; i < n; ++i) {
        groups[y(i)].push_back(i);
    }
    const int k = static_cast<int>(groups.size());
    if (k < 2) {
        throw std::invalid_argument(
            "f_classif requires at least 2 distinct classes.");
    }

    // Overall column-wise mean
    RowVec grand_mean = X.colwise().mean();

    // Between-group SS and within-group SS, per feature
    RowVec ss_between = RowVec::Zero(p);
    RowVec ss_within = RowVec::Zero(p);

    for (const auto& [_, idx] : groups) {
        Eigen::Index ng = static_cast<Eigen::Index>(idx.size());
        RowVec gmean = RowVec::Zero(p);
        for (Eigen::Index i : idx) gmean += X.row(i);
        gmean /= static_cast<Scalar>(ng);

        RowVec diff = gmean - grand_mean;
        ss_between += static_cast<Scalar>(ng) * diff.array().square().matrix();

        for (Eigen::Index i : idx) {
            RowVec d = X.row(i) - gmean;
            ss_within += d.array().square().matrix();
        }
    }

    Scalar dfb = static_cast<Scalar>(k - 1);
    Scalar dfw = static_cast<Scalar>(n - k);

    RowVec msb = ss_between / dfb;
    RowVec msw = ss_within / dfw;

    RowVec F(p);
    RowVec pv(p);
    for (Eigen::Index j = 0; j < p; ++j) {
        Scalar denom = msw(j);
        Scalar fval;
        if (denom <= std::numeric_limits<Scalar>::min()) {
            // Constant column → sklearn produces NaN
            fval = std::numeric_limits<Scalar>::quiet_NaN();
            F(j) = fval;
            pv(j) = std::numeric_limits<Scalar>::quiet_NaN();
        } else {
            fval = msb(j) / denom;
            F(j) = fval;
            pv(j) = detail::f_sf(fval, dfb, dfw);
        }
    }
    return {F, pv};
}

// ---------------------------------------------------------------------------
// f_regression — F-test on the correlation between feature and target.
// ---------------------------------------------------------------------------
/// @brief Univariate linear regression F-tests.
///
/// Mirrors
/// [sklearn.feature_selection.f_regression](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html).
///
/// @param X Design matrix of shape (n_samples, n_features).
/// @param y Continuous target of shape (n_samples,).
/// @param center If `true` (default), the data is centered before computing
///   the correlation.
/// @return Pair `(F, p)` per feature.
template <typename Scalar>
std::pair<Eigen::Matrix<Scalar, 1, Eigen::Dynamic>,
          Eigen::Matrix<Scalar, 1, Eigen::Dynamic>>
f_regression(const Eigen::Ref<const Eigen::Matrix<
                 Scalar, Eigen::Dynamic, Eigen::Dynamic>>& X,
             const Eigen::Ref<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>&
                 y,
             bool center = true) {
    using RowVec = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;
    using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    internal::check_non_empty(X);
    internal::check_consistent_length(X, y);

    const Eigen::Index n = X.rows();
    const Eigen::Index p = X.cols();

    Mat Xc;
    Vec yc;
    if (center) {
        RowVec xm = X.colwise().mean();
        Scalar ym = y.mean();
        Xc = X.rowwise() - xm;
        yc = y.array() - ym;
    } else {
        Xc = X;
        yc = y;
    }

    // Pearson correlation per column: r_j = (X_j · y) / (||X_j|| ||y||)
    Scalar y_norm = yc.norm();
    RowVec corr(p);
    for (Eigen::Index j = 0; j < p; ++j) {
        Scalar xn = Xc.col(j).norm();
        if (xn <= std::numeric_limits<Scalar>::min() ||
            y_norm <= std::numeric_limits<Scalar>::min()) {
            corr(j) = Scalar{0};
        } else {
            corr(j) = Xc.col(j).dot(yc) / (xn * y_norm);
        }
    }

    Scalar deg = static_cast<Scalar>(center ? (n - 2) : (n - 1));
    if (deg <= Scalar{0}) {
        throw std::invalid_argument(
            "f_regression: not enough samples for the requested centering.");
    }

    RowVec F(p);
    RowVec pv(p);
    for (Eigen::Index j = 0; j < p; ++j) {
        Scalar r2 = corr(j) * corr(j);
        Scalar denom = Scalar{1} - r2;
        Scalar fval;
        if (denom <= std::numeric_limits<Scalar>::min()) {
            // Perfect correlation
            fval = std::numeric_limits<Scalar>::infinity();
            F(j) = fval;
            pv(j) = Scalar{0};
        } else {
            fval = r2 * deg / denom;
            F(j) = fval;
            pv(j) = detail::f_sf(fval, Scalar{1}, deg);
        }
    }
    return {F, pv};
}

// ---------------------------------------------------------------------------
// chi2 — Chi-squared test between non-negative features and class labels.
// ---------------------------------------------------------------------------
/// @brief Compute chi-squared stats between each non-negative feature and class.
///
/// Mirrors
/// [sklearn.feature_selection.chi2](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html).
///
/// @param X Sample matrix of shape (n_samples, n_features). All entries must
///   be non-negative — otherwise a `std::invalid_argument` is thrown to match
///   sklearn behavior.
/// @param y Integer class labels of shape (n_samples,).
/// @return Pair `(chi2_stat, p)` per feature.
template <typename Scalar>
std::pair<Eigen::Matrix<Scalar, 1, Eigen::Dynamic>,
          Eigen::Matrix<Scalar, 1, Eigen::Dynamic>>
chi2(const Eigen::Ref<const Eigen::Matrix<
         Scalar, Eigen::Dynamic, Eigen::Dynamic>>& X,
     const Eigen::Ref<const Eigen::VectorXi>& y) {
    using RowVec = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;
    using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    internal::check_non_empty(X);
    internal::check_consistent_length(X, y);

    if ((X.array() < Scalar{0}).any()) {
        throw std::invalid_argument(
            "Input X must be non-negative for chi2 feature selection.");
    }

    const Eigen::Index n = X.rows();
    const Eigen::Index p = X.cols();

    // Build class index map
    std::map<int, int> class_to_row;
    for (Eigen::Index i = 0; i < n; ++i) {
        if (class_to_row.find(y(i)) == class_to_row.end()) {
            int next = static_cast<int>(class_to_row.size());
            class_to_row[y(i)] = next;
        }
    }
    const int k = static_cast<int>(class_to_row.size());
    if (k < 2) {
        throw std::invalid_argument(
            "chi2 requires at least 2 distinct classes.");
    }

    // observed[c, j] = sum of X[i, j] over i belonging to class c.
    Mat observed = Mat::Zero(k, p);
    Eigen::VectorX<Scalar> class_count = Eigen::VectorX<Scalar>::Zero(k);
    for (Eigen::Index i = 0; i < n; ++i) {
        int c = class_to_row[y(i)];
        observed.row(c) += X.row(i);
        class_count(c) += Scalar{1};
    }

    RowVec feature_total = X.colwise().sum();
    Scalar grand_total = feature_total.sum();

    // expected[c, j] = class_count[c] * feature_total[j] / grand_total
    // chi2_stat per feature = sum_c (obs - exp)^2 / exp
    RowVec stat(p);
    RowVec pv(p);
    for (Eigen::Index j = 0; j < p; ++j) {
        Scalar s = Scalar{0};
        if (feature_total(j) > Scalar{0} && grand_total > Scalar{0}) {
            for (int c = 0; c < k; ++c) {
                Scalar exp_cj = class_count(c) * feature_total(j) / grand_total;
                if (exp_cj > Scalar{0}) {
                    Scalar diff = observed(c, j) - exp_cj;
                    s += diff * diff / exp_cj;
                }
            }
        }
        stat(j) = s;
        pv(j) = detail::chi2_sf(s, static_cast<Scalar>(k - 1));
    }
    return {stat, pv};
}

// ---------------------------------------------------------------------------
// chi2 — sparse overload (v1.1.0 §3.2)
//
// Operates directly on the CSC nonzeros: per-column sums and per-class
// per-column observed totals are accumulated by iterating column-wise
// without materialising X dense. Implicit zeros contribute 0 to every
// expected/observed cell, matching the dense formula exactly.
// ---------------------------------------------------------------------------
template <typename Scalar, int Options, typename StorageIndex>
std::pair<Eigen::Matrix<Scalar, 1, Eigen::Dynamic>,
          Eigen::Matrix<Scalar, 1, Eigen::Dynamic>>
chi2(const Eigen::SparseMatrix<Scalar, Options, StorageIndex>& X,
     const Eigen::Ref<const Eigen::VectorXi>& y) {
    using RowVec = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;
    using Mat    = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using ColSparse =
        Eigen::SparseMatrix<Scalar, Eigen::ColMajor, StorageIndex>;

    if (X.rows() == 0 || X.cols() == 0) {
        throw std::invalid_argument("chi2: empty sparse matrix.");
    }
    if (X.rows() != y.size()) {
        throw std::invalid_argument(
            "chi2: X has " + std::to_string(X.rows()) +
            " rows but y has " + std::to_string(y.size()) + " entries.");
    }

    const ColSparse Xc = X;
    const Eigen::Index n = Xc.rows();
    const Eigen::Index p = Xc.cols();

    // Build class-index map (sorted ascending).
    std::map<int, int> class_to_row;
    for (Eigen::Index i = 0; i < n; ++i) {
        if (class_to_row.find(y(i)) == class_to_row.end()) {
            const int next = static_cast<int>(class_to_row.size());
            class_to_row[y(i)] = next;
        }
    }
    const int k = static_cast<int>(class_to_row.size());
    if (k < 2) {
        throw std::invalid_argument(
            "chi2 requires at least 2 distinct classes.");
    }

    Eigen::VectorX<Scalar> class_count = Eigen::VectorX<Scalar>::Zero(k);
    for (Eigen::Index i = 0; i < n; ++i) {
        class_count(class_to_row[y(i)]) += Scalar{1};
    }

    Mat observed = Mat::Zero(k, p);
    RowVec feature_total = RowVec::Zero(p);

    // One CSC column-iteration pass: accumulate observed[c, j] and
    // feature_total[j] from explicit nonzeros only. Reject negative values.
    for (Eigen::Index j = 0; j < p; ++j) {
        for (typename ColSparse::InnerIterator it(Xc, j); it; ++it) {
            const Scalar v = it.value();
            if (v < Scalar{0}) {
                throw std::invalid_argument(
                    "Input X must be non-negative for chi2 feature selection.");
            }
            const int c = class_to_row[y(it.row())];
            observed(c, j)    += v;
            feature_total(j)  += v;
        }
    }

    Scalar grand_total = feature_total.sum();
    RowVec stat(p);
    RowVec pv(p);
    for (Eigen::Index j = 0; j < p; ++j) {
        Scalar s = Scalar{0};
        if (feature_total(j) > Scalar{0} && grand_total > Scalar{0}) {
            for (int c = 0; c < k; ++c) {
                const Scalar exp_cj =
                    class_count(c) * feature_total(j) / grand_total;
                if (exp_cj > Scalar{0}) {
                    const Scalar diff = observed(c, j) - exp_cj;
                    s += diff * diff / exp_cj;
                }
            }
        }
        stat(j) = s;
        pv(j) = detail::chi2_sf(s, static_cast<Scalar>(k - 1));
    }
    return {stat, pv};
}

}  // namespace feature_selection
}  // namespace Skigen

#endif  // SKIGEN_FEATURE_SELECTION_SCORE_FUNCTIONS_H
