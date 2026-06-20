// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_CALIBRATION_CALIBRATED_CLASSIFIER_CV_H
#define SKIGEN_CALIBRATION_CALIBRATED_CLASSIFIER_CV_H

#include "../Core/Base.h"
#include "../Core/Validation.h"
#include "../Isotonic/IsotonicRegression.h"
#include "PlattScaling.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace Skigen {

/// @defgroup Algo_CalibratedClassifierCV Probability Calibration
/// @ingroup Calibration
/// @brief Cross-validated probability calibration via Platt sigmoid or
///   isotonic regression.
/// @{

/// @brief Calibration method for `CalibratedClassifierCV`.
enum class CalibrationMethod {
    Sigmoid,   ///< Platt's sigmoid calibration (logistic).
    Isotonic   ///< Non-parametric isotonic regression calibration.
};

/// @brief Probability calibration for a base classifier via cross-validation.
///
/// Mirrors the dense classification core of
/// [sklearn.calibration.CalibratedClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)
/// for an estimator that exposes `decision_function(X)` or `predict_proba(X)`.
///
/// Splits the training set into `cv` folds. For each fold the base
/// classifier is cloned, fit on the train portion, and applied to the
/// validation portion to produce raw positive-class probabilities; a
/// calibrator (sigmoid or isotonic) is then fit on `(p_raw, y)` for
/// that fold. At prediction time, all `cv` (base, calibrator) pairs are
/// applied and their calibrated probabilities are averaged
/// (`ensemble=true`, the sklearn 1.7 default). With `ensemble=false`,
/// the out-of-fold predictions fit one calibrator and the base estimator is
/// refit on the full training data.
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default |
/// |---|---|---|
/// | `estimator` | `Base` | (no default; required) |
/// | `method` | `CalibrationMethod` | `Sigmoid` |
/// | `cv` | `int` | `5` |
/// | `n_jobs` | `int` | `1` *(no-op for now; not implemented)* |
/// | `ensemble` | `bool` | `true` |
/// | `random_state` | `optional<uint64_t>` | `nullopt` |
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type |
/// |---|---|
/// | `classes()` | `Eigen::VectorXi` (length 2) |
/// | `n_classes()` | `int` |
/// | `n_estimators_fitted()` | `int` (== `cv` when `ensemble=true`) |
///
/// ### Limitations relative to scikit-learn `n_jobs > 1` is not honoured. The base estimator should expose
///   either `decision_function(X)` (preferred, used directly as calibration
///   input) or `predict_proba(X)` (logit-transformed before calibration),
///   matching sklearn's `_get_response_method` dispatch.
template <typename Base, typename Scalar = double>
class CalibratedClassifierCV
    : public Classifier<CalibratedClassifierCV<Base, Scalar>, Scalar> {
public:
    using ThisType   = CalibratedClassifierCV<Base, Scalar>;
    using BaseClass  = Classifier<ThisType, Scalar>;
    using typename BaseClass::MatrixType;
    using typename BaseClass::RowVectorType;
    using typename BaseClass::IndexType;
    using typename BaseClass::ScalarType;
    using typename BaseClass::LabelType;
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    /// @cond INTERNAL
    struct FoldFit {
        Base base;
        // Used when method == Sigmoid.
        VectorType A;
        VectorType B;
        // Used when method == Isotonic.
        std::vector<std::optional<IsotonicRegression<Scalar>>> iso;
    };
    /// @endcond

    explicit CalibratedClassifierCV(
        Base estimator,
        CalibrationMethod method = CalibrationMethod::Sigmoid,
        int cv = 5,
        int n_jobs = 1,
        bool ensemble = true,
        std::optional<uint64_t> random_state = std::nullopt)
        : estimator_(std::move(estimator)),
          method_(method),
          cv_(cv),
          n_jobs_(n_jobs),
          ensemble_(ensemble),
          random_state_(random_state) {
        if (cv_ < 2) {
            throw std::invalid_argument(
                "CalibratedClassifierCV: cv must be >= 2; got " +
                std::to_string(cv_));
        }
    }

    // -- Accessors ----------------------------------------------------------

    [[nodiscard]] CalibrationMethod method() const noexcept { return method_; }
    [[nodiscard]] int cv() const noexcept { return cv_; }
    [[nodiscard]] bool ensemble() const noexcept { return ensemble_; }

    [[nodiscard]] const Eigen::VectorXi& classes() const {
        this->check_is_fitted(); return classes_;
    }
    [[nodiscard]] int n_classes() const {
        this->check_is_fitted(); return static_cast<int>(classes_.size());
    }
    [[nodiscard]] int n_estimators_fitted() const {
        this->check_is_fitted();
        return ensemble_ ? static_cast<int>(folds_.size()) : 1;
    }
    [[nodiscard]] const std::vector<FoldFit>& folds() const {
        this->check_is_fitted(); return folds_;
    }

    SKIGEN_PARAMS((cv, cv_, int),
                  (ensemble, ensemble_, bool))

    // -- Fit/Predict --------------------------------------------------------

    CalibratedClassifierCV& fit_impl(
        const Eigen::Ref<const MatrixType>& X,
        const Eigen::Ref<const Eigen::VectorXi>& y) {
        internal::check_non_empty(X);
        internal::check_consistent_length(X, y);
        this->n_features_in_ = X.cols();

        // Discover unique class labels (sorted ascending).
        std::vector<int> uniq;
        uniq.reserve(static_cast<std::size_t>(y.size()));
        for (Eigen::Index i = 0; i < y.size(); ++i) uniq.push_back(y(i));
        std::sort(uniq.begin(), uniq.end());
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
        if (uniq.size() < 2) {
            throw std::invalid_argument(
                "CalibratedClassifierCV: at least two classes are required; got " +
                std::to_string(uniq.size()) + " classes.");
        }
        classes_ = Eigen::VectorXi(static_cast<Eigen::Index>(uniq.size()));
        for (Eigen::Index i = 0; i < classes_.size(); ++i) {
            classes_(i) = uniq[static_cast<std::size_t>(i)];
        }
        const Eigen::Index C = classes_.size();

        // Build a deterministic shuffle for the K-fold split.
        const Eigen::Index n = X.rows();
        std::vector<Eigen::Index> order(static_cast<std::size_t>(n));
        std::iota(order.begin(), order.end(), Eigen::Index{0});
        const uint64_t seed = random_state_.value_or(0ULL);
        std::mt19937_64 rng(seed);
        std::shuffle(order.begin(), order.end(), rng);

        // Compute fold sizes (matches sklearn.KFold's allocation).
        const int K = cv_;
        std::vector<int> fold_sizes(K, static_cast<int>(n) / K);
        for (int r = 0; r < static_cast<int>(n) % K; ++r) fold_sizes[r] += 1;

        folds_.clear();
        single_fit_.reset();
        if (ensemble_) folds_.reserve(static_cast<std::size_t>(K));
        MatrixType calibration_scores(n, C);
        MatrixType calibration_targets(n, C);

        Eigen::Index cursor = 0;
        for (int k = 0; k < K; ++k) {
            const int val_size = fold_sizes[k];

            // Validation indices for this fold.
            std::vector<Eigen::Index> val_idx;
            val_idx.reserve(static_cast<std::size_t>(val_size));
            for (int i = 0; i < val_size; ++i) {
                val_idx.push_back(order[static_cast<std::size_t>(cursor + i)]);
            }
            // Train indices = everything except the validation slice.
            std::vector<Eigen::Index> train_idx;
            train_idx.reserve(static_cast<std::size_t>(n - val_size));
            for (Eigen::Index j = 0; j < cursor; ++j) {
                train_idx.push_back(order[static_cast<std::size_t>(j)]);
            }
            for (Eigen::Index j = cursor + val_size; j < n; ++j) {
                train_idx.push_back(order[static_cast<std::size_t>(j)]);
            }
            cursor += val_size;

            // Materialise the train fold and fit a clone of the base.
            MatrixType X_train(static_cast<Eigen::Index>(train_idx.size()),
                               X.cols());
            Eigen::VectorXi y_train(
                static_cast<Eigen::Index>(train_idx.size()));
            for (std::size_t i = 0; i < train_idx.size(); ++i) {
                X_train.row(static_cast<Eigen::Index>(i)) =
                    X.row(train_idx[i]);
                y_train(static_cast<Eigen::Index>(i)) = y(train_idx[i]);
            }

            FoldFit ff{estimator_, VectorType::Zero(C), VectorType::Zero(C), {}};
            ff.iso.resize(static_cast<std::size_t>(C));
            ff.base.fit(X_train, y_train);

            // Get raw class probabilities on the validation rows.
            MatrixType X_val(static_cast<Eigen::Index>(val_idx.size()),
                             X.cols());
            MatrixType y_val_one_vs_rest(
                static_cast<Eigen::Index>(val_idx.size()), C);
            for (std::size_t i = 0; i < val_idx.size(); ++i) {
                X_val.row(static_cast<Eigen::Index>(i)) = X.row(val_idx[i]);
                for (Eigen::Index c = 0; c < C; ++c) {
                    y_val_one_vs_rest(static_cast<Eigen::Index>(i), c) =
                        (y(val_idx[i]) == classes_(c)) ? Scalar{1} : Scalar{0};
                }
            }
            MatrixType P = ff.base.predict_proba(X_val);
            // Expect one probability column per discovered class.
            if (P.cols() != C) {
                throw std::runtime_error(
                    "CalibratedClassifierCV: base estimator's "
                    "predict_proba must return one column per class; got " +
                    std::to_string(P.cols()) + " columns.");
            }
            MatrixType response_scores = get_score_matrix(ff.base, X_val, P, C);

            switch (method_) {
                case CalibrationMethod::Sigmoid: {
                    for (Eigen::Index c = 0; c < C; ++c) {
                        VectorType s = response_scores.col(c);
                        VectorType target = y_val_one_vs_rest.col(c);
                        if (!ensemble_) {
                            for (std::size_t i = 0; i < val_idx.size(); ++i) {
                                calibration_scores(val_idx[i], c) = s(static_cast<Eigen::Index>(i));
                                calibration_targets(val_idx[i], c) = target(static_cast<Eigen::Index>(i));
                            }
                        }
                        internal::fit_platt_sigmoid<Scalar>(s, target, ff.A(c), ff.B(c));
                    }
                    break;
                }
                case CalibrationMethod::Isotonic: {
                    for (Eigen::Index c = 0; c < C; ++c) {
                        VectorType p_class = P.col(c);
                        VectorType target = y_val_one_vs_rest.col(c);
                        if (!ensemble_) {
                            for (std::size_t i = 0; i < val_idx.size(); ++i) {
                                calibration_scores(val_idx[i], c) = p_class(static_cast<Eigen::Index>(i));
                                calibration_targets(val_idx[i], c) = target(static_cast<Eigen::Index>(i));
                            }
                        }
                        MatrixType S(p_class.size(), 1);
                        S.col(0) = p_class;
                        IsotonicRegression<Scalar> iso(
                            std::optional<Scalar>(Scalar{0}),
                            std::optional<Scalar>(Scalar{1}),
                            IsotonicIncreasing::True,
                            OutOfBounds::Clip);
                        iso.fit(S, target);
                        ff.iso[static_cast<std::size_t>(c)] = std::move(iso);
                    }
                    break;
                }
            }
            if (ensemble_) folds_.push_back(std::move(ff));
        }

        if (!ensemble_) {
            FoldFit ff{estimator_, VectorType::Zero(C), VectorType::Zero(C), {}};
            ff.iso.resize(static_cast<std::size_t>(C));
            ff.base.fit(X, y);
            switch (method_) {
                case CalibrationMethod::Sigmoid:
                    for (Eigen::Index c = 0; c < C; ++c) {
                        VectorType scores = calibration_scores.col(c);
                        VectorType targets = calibration_targets.col(c);
                        internal::fit_platt_sigmoid<Scalar>(scores, targets, ff.A(c), ff.B(c));
                    }
                    break;
                case CalibrationMethod::Isotonic: {
                    for (Eigen::Index c = 0; c < C; ++c) {
                        MatrixType S(calibration_scores.rows(), 1);
                        S.col(0) = calibration_scores.col(c);
                        VectorType targets = calibration_targets.col(c);
                        IsotonicRegression<Scalar> iso(
                            std::optional<Scalar>(Scalar{0}),
                            std::optional<Scalar>(Scalar{1}),
                            IsotonicIncreasing::True,
                            OutOfBounds::Clip);
                        iso.fit(S, targets);
                        ff.iso[static_cast<std::size_t>(c)] = std::move(iso);
                    }
                    break;
                }
            }
            single_fit_ = std::move(ff);
        }

        this->fitted_ = true;
        return *this;
    }

    [[nodiscard]] LabelType predict_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        MatrixType P = predict_proba(X);
        LabelType out(X.rows());
        for (Eigen::Index i = 0; i < X.rows(); ++i) {
            Eigen::Index best = 0;
            P.row(i).maxCoeff(&best);
            out(i) = classes_(best);
        }
        return out;
    }

    /// @brief Return averaged calibrated class probabilities, shape
    ///   (n_samples, n_classes).
    [[nodiscard]] MatrixType predict_proba(
        const Eigen::Ref<const MatrixType>& X) const {
        this->check_is_fitted();
        this->validate_feature_count(X);
        if (!ensemble_) {
            return normalized_probabilities(calibrated_probability_matrix(*single_fit_, X));
        }

        MatrixType acc = MatrixType::Zero(X.rows(), classes_.size());

        for (const auto& ff : folds_) {
            acc += normalized_probabilities(calibrated_probability_matrix(ff, X));
        }

        const Scalar inv_k = Scalar{1} / static_cast<Scalar>(folds_.size());
        acc *= inv_k;
        return acc;
    }

private:
    [[nodiscard]] MatrixType calibrated_probability_matrix(
        const FoldFit& ff,
        const Eigen::Ref<const MatrixType>& X) const {
        MatrixType out(X.rows(), classes_.size());
        switch (method_) {
            case CalibrationMethod::Sigmoid: {
                MatrixType scores = get_score_matrix(ff.base, X, ff.base.predict_proba(X), classes_.size());
                for (Eigen::Index c = 0; c < classes_.size(); ++c) {
                    out.col(c) = internal::apply_platt_sigmoid<Scalar>(
                        scores.col(c), ff.A(c), ff.B(c));
                }
                return out;
            }
            case CalibrationMethod::Isotonic: {
                MatrixType P = ff.base.predict_proba(X);
                for (Eigen::Index c = 0; c < classes_.size(); ++c) {
                    MatrixType S(P.rows(), 1);
                    S.col(0) = P.col(c);
                    out.col(c) = ff.iso[static_cast<std::size_t>(c)]->predict(S);
                }
                return out;
            }
        }
        return MatrixType::Zero(X.rows(), classes_.size());
    }

    [[nodiscard]] static MatrixType normalized_probabilities(MatrixType P) {
        for (Eigen::Index i = 0; i < P.rows(); ++i) {
            for (Eigen::Index j = 0; j < P.cols(); ++j) {
                if (!std::isfinite(P(i, j)) || P(i, j) < Scalar{0}) P(i, j) = Scalar{0};
            }
            Scalar sum = P.row(i).sum();
            if (sum <= std::numeric_limits<Scalar>::epsilon()) {
                P.row(i).setConstant(Scalar{1} / static_cast<Scalar>(P.cols()));
            } else {
                P.row(i) /= sum;
            }
        }
        return P;
    }

    static MatrixType get_score_matrix(const Base& est,
                                       const Eigen::Ref<const MatrixType>& X,
                                       const MatrixType& probabilities,
                                       Eigen::Index n_classes) {
        if constexpr (requires(const Base& b, const MatrixType& m) {
                          { b.decision_function(m) };
                      }) {
            auto df = est.decision_function(X);
            if constexpr (requires { df.col(0); }) {
                if (df.cols() == n_classes) return df;
                if (df.cols() == 1 && n_classes == 2) {
                    MatrixType out(df.rows(), 2);
                    out.col(0) = -df.col(0);
                    out.col(1) = df.col(0);
                    return out;
                }
                return df.leftCols(n_classes);
            } else {
                MatrixType out(df.size(), n_classes);
                if (n_classes == 2) {
                    out.col(0) = -df;
                    out.col(1) = df;
                } else {
                    out.setZero();
                    out.col(0) = df;
                }
                return out;
            }
        } else {
            MatrixType scores(probabilities.rows(), probabilities.cols());
            for (Eigen::Index i = 0; i < probabilities.rows(); ++i) {
                for (Eigen::Index c = 0; c < probabilities.cols(); ++c) {
                    const Scalar p = std::clamp(
                        probabilities(i, c),
                        std::numeric_limits<Scalar>::epsilon(),
                        Scalar{1} - std::numeric_limits<Scalar>::epsilon());
                    scores(i, c) = std::log(p / (Scalar{1} - p));
                }
            }
            return scores;
        }
    }

    Base estimator_;                 // user-supplied prototype (cloned per fold)
    CalibrationMethod method_;
    int cv_;
    int n_jobs_;                     // currently no-op
    bool ensemble_;
    std::optional<uint64_t> random_state_;

    Eigen::VectorXi classes_;
    std::vector<FoldFit> folds_;
    std::optional<FoldFit> single_fit_;
};

/// @}

} // namespace Skigen

#endif // SKIGEN_CALIBRATION_CALIBRATED_CLASSIFIER_CV_H
