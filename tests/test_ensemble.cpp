// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#include <Skigen/Dense>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Minimal test harness (mirrors tests/test_standard_scaler.cpp)
// ---------------------------------------------------------------------------

static int g_passed = 0;
static int g_failed = 0;

struct TestFailure : std::exception {
    std::string msg;
    TestFailure(std::string m) : msg(std::move(m)) {}
    const char* what() const noexcept override { return msg.c_str(); }
};

#define ASSERT_TRUE(cond)                                                     \
    do {                                                                       \
        if (!(cond)) {                                                         \
            throw TestFailure(std::string(__FILE__) + ":" +                    \
                              std::to_string(__LINE__) + ": ASSERT_TRUE(" +    \
                              #cond + ") failed");                             \
        }                                                                      \
    } while (false)

#define ASSERT_THROW(expr, ExType)                                            \
    do {                                                                       \
        bool caught = false;                                                   \
        try { static_cast<void>(expr); } catch (const ExType&) { caught = true; } \
        if (!caught) {                                                         \
            throw TestFailure(std::string(__FILE__) + ":" +                    \
                              std::to_string(__LINE__) +                       \
                              ": Expected exception " #ExType);                \
        }                                                                      \
    } while (false)

template <typename Scalar>
void assert_near(Scalar a, Scalar b, Scalar tol, const char* file, int line) {
    if (std::abs(a - b) > tol) {
        std::ostringstream oss;
        oss << file << ":" << line << ": ASSERT_NEAR failed: " << a
            << " vs " << b << " (tol=" << tol << ")";
        throw TestFailure(oss.str());
    }
}

#define ASSERT_NEAR(a, b, tol) assert_near(a, b, tol, __FILE__, __LINE__)

static void run_test(const std::string& name, std::function<void()> fn) {
    try {
        fn();
        ++g_passed;
        std::cout << "  PASS  " << name << "\n";
    } catch (const TestFailure& e) {
        ++g_failed;
        std::cout << "  FAIL  " << name << "\n        " << e.what() << "\n";
    } catch (const std::exception& e) {
        ++g_failed;
        std::cout << "  FAIL  " << name << "\n        exception: " << e.what()
                  << "\n";
    }
}

// ---------------------------------------------------------------------------
// Helpers — synthetic datasets
// ---------------------------------------------------------------------------

static void make_two_class_linear(int n, Eigen::MatrixXd& X, Eigen::VectorXi& y,
                                  uint64_t seed = 7) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> noise(0.0, 0.4);
    X.resize(n, 2);
    y.resize(n);
    for (int i = 0; i < n; ++i) {
        bool pos = (i % 2 == 0);
        X(i, 0) = (pos ? 1.0 : -1.0) + noise(rng);
        X(i, 1) = (pos ? 1.0 : -1.0) + noise(rng);
        y(i) = pos ? 1 : 0;
    }
}

static void make_three_class_xor(int n_per, Eigen::MatrixXd& X, Eigen::VectorXi& y,
                                 uint64_t seed = 11) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> noise(0.0, 0.25);
    int n = 3 * n_per;
    X.resize(n, 2);
    y.resize(n);
    for (int i = 0; i < n_per; ++i) {
        X(i, 0) = 1.0 + noise(rng); X(i, 1) = 1.0 + noise(rng); y(i) = 0;
    }
    for (int i = 0; i < n_per; ++i) {
        X(n_per + i, 0) = -1.0 + noise(rng);
        X(n_per + i, 1) = -1.0 + noise(rng); y(n_per + i) = 1;
    }
    for (int i = 0; i < n_per; ++i) {
        X(2 * n_per + i, 0) = 1.0 + noise(rng);
        X(2 * n_per + i, 1) = -1.0 + noise(rng); y(2 * n_per + i) = 2;
    }
}

// ---------------------------------------------------------------------------
// RandomForestClassifier tests
// ---------------------------------------------------------------------------

void test_rfc_two_class_linear() {
    Eigen::MatrixXd X; Eigen::VectorXi y;
    make_two_class_linear(60, X, y, 1);

    Skigen::RandomForestClassifier<double> rf(
        20, Skigen::RandomForestClassifier<double>::CriterionClf::Gini,
        std::nullopt, 2, 1, 0.0,
        Skigen::RandomForestClassifier<double>::MaxFeaturesMode::Sqrt,
        std::nullopt, std::nullopt, 0.0, true, false, 1,
        std::optional<uint64_t>(42));
    rf.fit(X, y);

    double acc = rf.score(X, y);
    ASSERT_TRUE(acc >= 0.95);
    ASSERT_TRUE(static_cast<int>(rf.estimators().size()) == 20);
}

void test_rfc_three_class_xor() {
    Eigen::MatrixXd X; Eigen::VectorXi y;
    make_three_class_xor(40, X, y, 5);

    Skigen::RandomForestClassifier<double> rf(
        30, Skigen::RandomForestClassifier<double>::CriterionClf::Gini,
        std::nullopt, 2, 1, 0.0,
        Skigen::RandomForestClassifier<double>::MaxFeaturesMode::Sqrt,
        std::nullopt, std::nullopt, 0.0, true, false, 1,
        std::optional<uint64_t>(42));
    rf.fit(X, y);

    auto pred = rf.predict(X);
    int correct = 0;
    for (Eigen::Index i = 0; i < y.size(); ++i)
        if (pred(i) == y(i)) ++correct;
    double acc = static_cast<double>(correct) / static_cast<double>(y.size());
    ASSERT_TRUE(acc >= 0.85);
    ASSERT_TRUE(rf.n_classes() == 3);
    ASSERT_TRUE(rf.classes().size() == 3);
}

void test_rfc_predict_proba() {
    Eigen::MatrixXd X; Eigen::VectorXi y;
    make_three_class_xor(30, X, y, 9);

    Skigen::RandomForestClassifier<double> rf(
        15, Skigen::RandomForestClassifier<double>::CriterionClf::Gini,
        std::nullopt, 2, 1, 0.0,
        Skigen::RandomForestClassifier<double>::MaxFeaturesMode::Sqrt,
        std::nullopt, std::nullopt, 0.0, true, false, 1,
        std::optional<uint64_t>(123));
    rf.fit(X, y);
    auto P = rf.predict_proba(X);
    ASSERT_TRUE(P.rows() == X.rows());
    ASSERT_TRUE(P.cols() == 3);
    for (Eigen::Index i = 0; i < P.rows(); ++i) {
        double s = 0.0;
        for (Eigen::Index c = 0; c < P.cols(); ++c) {
            ASSERT_TRUE(P(i, c) >= -1e-12 && P(i, c) <= 1.0 + 1e-12);
            s += P(i, c);
        }
        ASSERT_NEAR(s, 1.0, 1e-9);
    }
}

void test_rfc_n_estimators_honoured() {
    Eigen::MatrixXd X; Eigen::VectorXi y;
    make_two_class_linear(40, X, y, 2);
    Skigen::RandomForestClassifier<double> rf(
        7, Skigen::RandomForestClassifier<double>::CriterionClf::Gini,
        std::nullopt, 2, 1, 0.0,
        Skigen::RandomForestClassifier<double>::MaxFeaturesMode::Sqrt,
        std::nullopt, std::nullopt, 0.0, true, false, 1,
        std::optional<uint64_t>(0));
    rf.fit(X, y);
    ASSERT_TRUE(static_cast<int>(rf.estimators().size()) == 7);
}

void test_rfc_feature_importances_normalised() {
    Eigen::MatrixXd X; Eigen::VectorXi y;
    make_two_class_linear(80, X, y, 3);

    Skigen::RandomForestClassifier<double> rf(
        20, Skigen::RandomForestClassifier<double>::CriterionClf::Gini,
        std::nullopt, 2, 1, 0.0,
        Skigen::RandomForestClassifier<double>::MaxFeaturesMode::All,
        std::nullopt, std::nullopt, 0.0, true, false, 1,
        std::optional<uint64_t>(2024));
    rf.fit(X, y);
    auto fi = rf.feature_importances();
    ASSERT_TRUE(fi.size() == 2);
    double s = 0.0;
    for (Eigen::Index i = 0; i < fi.size(); ++i) {
        ASSERT_TRUE(fi(i) >= -1e-12);
        s += fi(i);
    }
    ASSERT_NEAR(s, 1.0, 1e-6);
}

void test_rfc_oob_score_populated() {
    Eigen::MatrixXd X; Eigen::VectorXi y;
    make_two_class_linear(60, X, y, 4);
    Skigen::RandomForestClassifier<double> rf(
        25, Skigen::RandomForestClassifier<double>::CriterionClf::Gini,
        std::nullopt, 2, 1, 0.0,
        Skigen::RandomForestClassifier<double>::MaxFeaturesMode::Sqrt,
        std::nullopt, std::nullopt, 0.0, true, /*oob_score=*/true, 1,
        std::optional<uint64_t>(7));
    rf.fit(X, y);
    double oob = rf.oob_score();
    ASSERT_TRUE(oob >= 0.0 && oob <= 1.0);
    auto& dec = rf.oob_decision_function();
    ASSERT_TRUE(dec.rows() == X.rows());
    ASSERT_TRUE(dec.cols() == 2);
}

void test_rfc_random_state_reproducible() {
    Eigen::MatrixXd X; Eigen::VectorXi y;
    make_two_class_linear(50, X, y, 6);

    auto build = [&]() {
        Skigen::RandomForestClassifier<double> rf(
            10, Skigen::RandomForestClassifier<double>::CriterionClf::Gini,
            std::nullopt, 2, 1, 0.0,
            Skigen::RandomForestClassifier<double>::MaxFeaturesMode::Sqrt,
            std::nullopt, std::nullopt, 0.0, true, false, 1,
            std::optional<uint64_t>(99));
        rf.fit(X, y);
        return rf.predict(X);
    };
    auto p1 = build();
    auto p2 = build();
    ASSERT_TRUE(p1.size() == p2.size());
    for (Eigen::Index i = 0; i < p1.size(); ++i)
        ASSERT_TRUE(p1(i) == p2(i));
}

void test_rfc_bootstrap_false() {
    Eigen::MatrixXd X; Eigen::VectorXi y;
    make_two_class_linear(40, X, y, 8);
    Skigen::RandomForestClassifier<double> rf(
        5, Skigen::RandomForestClassifier<double>::CriterionClf::Gini,
        std::nullopt, 2, 1, 0.0,
        Skigen::RandomForestClassifier<double>::MaxFeaturesMode::All,
        std::nullopt, std::nullopt, 0.0,
        /*bootstrap=*/false, false, 1,
        std::optional<uint64_t>(0));
    rf.fit(X, y);
    // With bootstrap=false and All features, all trees see the same data.
    // They should all produce identical predictions.
    auto p0 = rf.estimators()[0].predict(X);
    for (std::size_t t = 1; t < rf.estimators().size(); ++t) {
        auto pt = rf.estimators()[t].predict(X);
        for (Eigen::Index i = 0; i < p0.size(); ++i)
            ASSERT_TRUE(p0(i) == pt(i));
    }
}

void test_rfc_n_jobs_equivalence() {
    Eigen::MatrixXd X; Eigen::VectorXi y;
    make_two_class_linear(40, X, y, 10);

    auto run = [&](int jobs) {
        Skigen::RandomForestClassifier<double> rf(
            8, Skigen::RandomForestClassifier<double>::CriterionClf::Gini,
            std::nullopt, 2, 1, 0.0,
            Skigen::RandomForestClassifier<double>::MaxFeaturesMode::Sqrt,
            std::nullopt, std::nullopt, 0.0, true, false, jobs,
            std::optional<uint64_t>(2026));
        rf.fit(X, y);
        return rf.predict(X);
    };
    auto p1 = run(1);
    auto p2 = run(2);
    for (Eigen::Index i = 0; i < p1.size(); ++i)
        ASSERT_TRUE(p1(i) == p2(i));
}

void test_rfc_unsupported_criterion_throws() {
    bool threw = false;
    try {
        Skigen::RandomForestClassifier<double> rf(
            10, Skigen::RandomForestClassifier<double>::CriterionClf::LogLoss);
        (void)rf;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_TRUE(threw);
}

void test_rfc_max_features_modes() {
    Eigen::MatrixXd X; Eigen::VectorXi y;
    make_two_class_linear(40, X, y, 14);
    using MF = Skigen::RandomForestClassifier<double>::MaxFeaturesMode;
    for (MF mode : {MF::All, MF::Sqrt, MF::Log2}) {
        Skigen::RandomForestClassifier<double> rf(
            5, Skigen::RandomForestClassifier<double>::CriterionClf::Gini,
            std::nullopt, 2, 1, 0.0, mode, std::nullopt, std::nullopt,
            0.0, true, false, 1, std::optional<uint64_t>(33));
        rf.fit(X, y);
        auto pred = rf.predict(X);
        ASSERT_TRUE(pred.size() == X.rows());
    }
}

void test_rfc_feature_count_mismatch_throws() {
    Eigen::MatrixXd X; Eigen::VectorXi y;
    make_two_class_linear(20, X, y, 0);
    Skigen::RandomForestClassifier<double> rf(
        5, Skigen::RandomForestClassifier<double>::CriterionClf::Gini,
        std::nullopt, 2, 1, 0.0,
        Skigen::RandomForestClassifier<double>::MaxFeaturesMode::Sqrt,
        std::nullopt, std::nullopt, 0.0, true, false, 1,
        std::optional<uint64_t>(1));
    rf.fit(X, y);
    Eigen::MatrixXd X_bad(3, 5);
    X_bad.setOnes();
    ASSERT_THROW(rf.predict(X_bad), std::invalid_argument);
}

void test_rfc_not_fitted_throws() {
    Skigen::RandomForestClassifier<double> rf(5);
    Eigen::MatrixXd X(3, 2);
    X.setOnes();
    ASSERT_THROW(rf.predict(X), std::runtime_error);
}

// ---------------------------------------------------------------------------
// RandomForestRegressor tests
// ---------------------------------------------------------------------------

void test_rfr_noisy_linear() {
    int n = 100;
    std::mt19937_64 rng(123);
    std::normal_distribution<double> noise(0.0, 0.05);
    Eigen::MatrixXd X(n, 1);
    Eigen::VectorXd y(n);
    for (int i = 0; i < n; ++i) {
        X(i, 0) = static_cast<double>(i) / static_cast<double>(n);
        y(i) = 2.0 * X(i, 0) + 1.0 + noise(rng);
    }
    Skigen::RandomForestRegressor<double> rf(
        30, Skigen::RandomForestRegressor<double>::CriterionReg::SquaredError,
        std::nullopt, 2, 1, 0.0,
        Skigen::RandomForestRegressor<double>::MaxFeaturesMode::All,
        std::nullopt, std::nullopt, 0.0, true, false, 1,
        std::optional<uint64_t>(42));
    rf.fit(X, y);
    double r2 = rf.score(X, y);
    ASSERT_TRUE(r2 > 0.85);
}

void test_rfr_predict_matches_mean_of_two_trees() {
    int n = 30;
    Eigen::MatrixXd X(n, 1);
    Eigen::VectorXd y(n);
    for (int i = 0; i < n; ++i) {
        X(i, 0) = static_cast<double>(i);
        y(i) = static_cast<double>(i) * 0.5;
    }
    Skigen::RandomForestRegressor<double> rf(
        2, Skigen::RandomForestRegressor<double>::CriterionReg::SquaredError,
        std::nullopt, 2, 1, 0.0,
        Skigen::RandomForestRegressor<double>::MaxFeaturesMode::All,
        std::nullopt, std::nullopt, 0.0, true, false, 1,
        std::optional<uint64_t>(15));
    rf.fit(X, y);

    auto p_forest = rf.predict(X);
    auto p_t0 = rf.estimators()[0].predict(X);
    auto p_t1 = rf.estimators()[1].predict(X);
    for (Eigen::Index i = 0; i < X.rows(); ++i) {
        double expected = 0.5 * (p_t0(i) + p_t1(i));
        ASSERT_NEAR(p_forest(i), expected, 1e-12);
    }
}

void test_rfr_unsupported_criterion_throws() {
    bool threw = false;
    try {
        Skigen::RandomForestRegressor<double> rf(
            5, Skigen::RandomForestRegressor<double>::CriterionReg::AbsoluteError);
        (void)rf;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_TRUE(threw);
}

void test_rfr_multi_output_predicts_two_targets() {
    constexpr int n = 60;
    Eigen::MatrixXd X(n, 1);
    Eigen::MatrixXd Y(n, 2);
    for (int i = 0; i < n; ++i) {
        X(i, 0) = static_cast<double>(i);
        Y(i, 0) = (i < n / 2) ? 1.0 : 5.0;
        Y(i, 1) = (i < n / 3) ? -2.0 : 4.0;
    }
    Skigen::RandomForestRegressor<double> rf(
        20, Skigen::RandomForestRegressor<double>::CriterionReg::SquaredError,
        std::nullopt, 2, 1, 0.0,
        Skigen::RandomForestRegressor<double>::MaxFeaturesMode::All,
        std::nullopt, std::nullopt, 0.0, true, false, 1,
        std::optional<uint64_t>(5));
    rf.fit_multi(X, Y);

    ASSERT_TRUE(rf.n_targets() == 2);
    Eigen::MatrixXd Yp = rf.predict_multi(X);
    ASSERT_TRUE(Yp.rows() == n);
    ASSERT_TRUE(Yp.cols() == 2);
    ASSERT_TRUE(Yp.allFinite());
    // First and last rows should clearly track the two target levels.
    ASSERT_TRUE(Yp(0, 0) < Yp(n - 1, 0));
    ASSERT_TRUE(Yp(0, 1) < Yp(n - 1, 1));
}

// ---------------------------------------------------------------------------
// GradientBoostingRegressor
// ---------------------------------------------------------------------------

void test_gbr_recovers_linear_signal() {
    constexpr int n = 200;
    Eigen::MatrixXd X(n, 1);
    Eigen::VectorXd y(n);
    for (int i = 0; i < n; ++i) {
        const double x = -1.0 + 2.0 * static_cast<double>(i) / (n - 1);
        X(i, 0) = x;
        y(i)    = 3.0 * x + 1.0;            // exact linear, no noise
    }

    using GBR = Skigen::GradientBoostingRegressor<double>;
    GBR gb(GBR::Loss::SquaredError, /*lr=*/0.1, /*n_estimators=*/200,
           /*subsample=*/1.0, GBR::CriterionGB::FriedmanMSE,
           2, 1, 0.0, /*max_depth=*/3);
    gb.fit(X, y);
    ASSERT_TRUE(gb.score(X, y) > 0.99);
}

void test_gbr_init_equals_mean_of_y() {
    Eigen::MatrixXd X(5, 1); X << 0, 1, 2, 3, 4;
    Eigen::VectorXd y(5);    y << 10, 12, 11, 13, 14;
    Skigen::GradientBoostingRegressor<double> gb;
    gb.fit(X, y);
    ASSERT_NEAR(gb.init(), y.mean(), 1e-12);
}

void test_gbr_n_estimators_honoured() {
    Eigen::MatrixXd X(20, 2);
    Eigen::VectorXd y(20);
    for (int i = 0; i < 20; ++i) {
        X(i, 0) = i; X(i, 1) = 20 - i;
        y(i)    = 0.5 * X(i, 0) - 0.3 * X(i, 1);
    }
    Skigen::GradientBoostingRegressor<double> gb(
        Skigen::GradientBoostingRegressor<double>::Loss::SquaredError,
        0.1, /*n_estimators=*/30);
    gb.fit(X, y);
    ASSERT_TRUE(gb.estimators().size() == 30);
    ASSERT_TRUE(gb.n_estimators_fitted() == 30);
}

void test_gbr_train_score_decreases() {
    Eigen::MatrixXd X(50, 1);
    Eigen::VectorXd y(50);
    for (int i = 0; i < 50; ++i) { X(i, 0) = i; y(i) = std::sin(0.2 * i); }
    Skigen::GradientBoostingRegressor<double> gb(
        Skigen::GradientBoostingRegressor<double>::Loss::SquaredError,
        0.1, /*n_estimators=*/50);
    gb.fit(X, y);
    const Eigen::VectorXd s = gb.train_score();
    ASSERT_TRUE(s.size() == 50);
    ASSERT_TRUE(s(0) > s(s.size() - 1));   // last stage MSE < first stage MSE
}

void test_gbr_feature_importances_shape_and_normalised() {
    // Independent features: only feature 0 carries signal; feature 1 is a
    // shuffled copy of [0..n) (so it is not collinear with feature 0);
    // feature 2 is constant zero (no information).
    constexpr int n = 60;
    Eigen::MatrixXd X(n, 3);
    Eigen::VectorXd y(n);
    std::mt19937 rng(123);
    std::vector<int> shuffled(n);
    std::iota(shuffled.begin(), shuffled.end(), 0);
    std::shuffle(shuffled.begin(), shuffled.end(), rng);
    for (int i = 0; i < n; ++i) {
        X(i, 0) = i;
        X(i, 1) = shuffled[i];
        X(i, 2) = 0.0;
        y(i)    = 2.0 * X(i, 0);          // signal lives only in feature 0
    }
    Skigen::GradientBoostingRegressor<double> gb(
        Skigen::GradientBoostingRegressor<double>::Loss::SquaredError,
        0.1, 50, 1.0,
        Skigen::GradientBoostingRegressor<double>::CriterionGB::FriedmanMSE,
        2, 1, 0.0, 3, 0.0, std::optional<uint64_t>(7));
    gb.fit(X, y);
    ASSERT_TRUE(gb.feature_importances().size() == 3);
    double sum = gb.feature_importances().sum();
    ASSERT_NEAR(sum, 1.0, 1e-9);
    // Feature 0 should dominate (no other feature has any predictive power).
    ASSERT_TRUE(gb.feature_importances()(0) > 0.5);
}

void test_gbr_absolute_error_recovers_signal() {
    using GBR = Skigen::GradientBoostingRegressor<double>;
    constexpr int n = 80;
    Eigen::MatrixXd X(n, 1);
    Eigen::VectorXd y(n);
    for (int i = 0; i < n; ++i) {
        X(i, 0) = static_cast<double>(i) / n;
        y(i) = 3.0 * X(i, 0) + 1.0;
    }
    GBR gb(GBR::Loss::AbsoluteError, 0.1, 100, 1.0,
           GBR::CriterionGB::FriedmanMSE, 2, 1, 0.0, 3, 0.0,
           std::optional<uint64_t>(0));
    gb.fit(X, y);
    ASSERT_TRUE(gb.score(X, y) > 0.95);
    ASSERT_TRUE(gb.train_score().allFinite());
    // Absolute-error train loss should decrease from first to last stage.
    ASSERT_TRUE(gb.train_score()(gb.n_estimators_fitted() - 1) <
                gb.train_score()(0));
}

void test_gbr_quantile_loss_tracks_quantile() {
    using GBR = Skigen::GradientBoostingRegressor<double>;
    constexpr int n = 200;
    Eigen::MatrixXd X(n, 1);
    Eigen::VectorXd y(n);
    std::mt19937_64 rng(7);
    std::normal_distribution<double> noise(0.0, 1.0);
    for (int i = 0; i < n; ++i) {
        X(i, 0) = static_cast<double>(i) / n;
        y(i) = 2.0 * X(i, 0) + noise(rng);
    }
    GBR high(GBR::Loss::Quantile, 0.1, 100, 1.0,
             GBR::CriterionGB::FriedmanMSE, 2, 1, 0.0, 3, 0.0,
             std::optional<uint64_t>(1), /*alpha=*/0.9);
    GBR low(GBR::Loss::Quantile, 0.1, 100, 1.0,
            GBR::CriterionGB::FriedmanMSE, 2, 1, 0.0, 3, 0.0,
            std::optional<uint64_t>(1), /*alpha=*/0.1);
    high.fit(X, y);
    low.fit(X, y);
    const Eigen::VectorXd hp = high.predict(X);
    const Eigen::VectorXd lp = low.predict(X);
    int above_high = 0, above_low = 0;
    for (int i = 0; i < n; ++i) {
        if (y(i) > hp(i)) ++above_high;
        if (y(i) > lp(i)) ++above_low;
    }
    // The 0.9 quantile should sit above most points; the 0.1 below most.
    ASSERT_TRUE(above_high < above_low);
    ASSERT_TRUE(above_high < n / 4);
    ASSERT_TRUE(above_low > 3 * n / 4);
}

void test_gbr_quantile_invalid_alpha_throws() {
    using GBR = Skigen::GradientBoostingRegressor<double>;
    bool threw = false;
    try {
        GBR gb(GBR::Loss::Quantile, 0.1, 100, 1.0,
               GBR::CriterionGB::FriedmanMSE, 2, 1, 0.0, 3, 0.0,
               std::nullopt, /*alpha=*/1.5);
        (void)gb;
    } catch (const std::invalid_argument&) { threw = true; }
    ASSERT_TRUE(threw);
}

void test_gbr_stochastic_subsample_fits_and_is_deterministic() {
    using GBR = Skigen::GradientBoostingRegressor<double>;
    constexpr int n = 120;
    Eigen::MatrixXd X(n, 1);
    Eigen::VectorXd y(n);
    std::mt19937_64 rng(3);
    std::normal_distribution<double> noise(0.0, 0.1);
    for (int i = 0; i < n; ++i) {
        X(i, 0) = static_cast<double>(i) / n;
        y(i) = 2.0 * X(i, 0) + noise(rng);
    }
    GBR a(GBR::Loss::SquaredError, 0.1, 80, /*subsample=*/0.6,
          GBR::CriterionGB::FriedmanMSE, 2, 1, 0.0, 3, 0.0,
          std::optional<uint64_t>(42));
    GBR b(GBR::Loss::SquaredError, 0.1, 80, /*subsample=*/0.6,
          GBR::CriterionGB::FriedmanMSE, 2, 1, 0.0, 3, 0.0,
          std::optional<uint64_t>(42));
    a.fit(X, y);
    b.fit(X, y);
    ASSERT_TRUE(a.score(X, y) > 0.9);
    // Same seed -> identical stochastic subsampling -> identical predictions.
    const Eigen::VectorXd pa = a.predict(X);
    const Eigen::VectorXd pb = b.predict(X);
    double max_diff = 0.0;
    for (int i = 0; i < pa.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(pa(i) - pb(i)));
    }
    ASSERT_TRUE(max_diff < 1e-12);
}

void test_gbr_invalid_subsample_throws() {
    using GBR = Skigen::GradientBoostingRegressor<double>;
    bool threw = false;
    try {
        GBR gb(GBR::Loss::SquaredError, 0.1, 100, /*subsample=*/0.0);
        (void)gb;
    } catch (const std::invalid_argument&) { threw = true; }
    ASSERT_TRUE(threw);
}

void test_gbr_not_fitted_throws() {
    Skigen::GradientBoostingRegressor<double> gb;
    Eigen::MatrixXd X(2, 1); X << 0.0, 1.0;
    bool threw = false;
    try { (void)gb.predict(X); }
    catch (const std::runtime_error&) { threw = true; }
    ASSERT_TRUE(threw);
}

// ---------------------------------------------------------------------------
// GradientBoostingClassifier (binary, log-loss)
// ---------------------------------------------------------------------------

void test_gbc_binary_separable_high_accuracy() {
    constexpr int n = 200;
    Eigen::MatrixXd X(n, 2);
    Eigen::VectorXi y(n);
    std::mt19937_64 rng(11);
    std::normal_distribution<double> ns(0.0, 0.5);
    for (int i = 0; i < n; ++i) {
        const double cls = (i < n / 2) ? -1.0 : 1.0;
        X(i, 0) = cls + ns(rng);
        X(i, 1) = cls + ns(rng);
        y(i)    = (cls > 0) ? 1 : 0;
    }
    Skigen::GradientBoostingClassifier<double> gb(
        Skigen::GradientBoostingClassifier<double>::Loss::LogLoss,
        0.1, /*n_estimators=*/100);
    gb.fit(X, y);
    auto preds = gb.predict(X);
    int correct = 0;
    for (int i = 0; i < n; ++i) if (preds(i) == y(i)) ++correct;
    const double acc = static_cast<double>(correct) / n;
    ASSERT_TRUE(acc > 0.95);
}

void test_gbc_predict_proba_shape_and_rows_sum_to_one() {
    constexpr int n = 30;
    Eigen::MatrixXd X(n, 1);
    Eigen::VectorXi y(n);
    for (int i = 0; i < n; ++i) {
        X(i, 0) = i;
        y(i)    = (i >= n / 2) ? 1 : 0;
    }
    Skigen::GradientBoostingClassifier<double> gb(
        Skigen::GradientBoostingClassifier<double>::Loss::LogLoss,
        0.1, 50);
    gb.fit(X, y);
    Eigen::MatrixXd P = gb.predict_proba(X);
    ASSERT_TRUE(P.rows() == n);
    ASSERT_TRUE(P.cols() == 2);
    for (int i = 0; i < n; ++i) {
        ASSERT_NEAR(P(i, 0) + P(i, 1), 1.0, 1e-12);
        ASSERT_TRUE(P(i, 0) >= 0.0 && P(i, 0) <= 1.0);
        ASSERT_TRUE(P(i, 1) >= 0.0 && P(i, 1) <= 1.0);
    }
}

void test_gbc_init_log_odds() {
    // 80% class 1, 20% class 0  ->  log(0.8/0.2) = log 4
    constexpr int n = 50;
    Eigen::MatrixXd X(n, 1);
    Eigen::VectorXi y(n);
    for (int i = 0; i < n; ++i) {
        X(i, 0) = i;
        y(i)    = (i < n * 4 / 5) ? 1 : 0;
    }
    Skigen::GradientBoostingClassifier<double> gb;
    gb.fit(X, y);
    ASSERT_NEAR(gb.init(), std::log(4.0), 1e-9);
}

void test_gbc_train_score_decreases() {
    constexpr int n = 40;
    Eigen::MatrixXd X(n, 1);
    Eigen::VectorXi y(n);
    for (int i = 0; i < n; ++i) {
        X(i, 0) = i;
        y(i)    = (i >= n / 2) ? 1 : 0;
    }
    Skigen::GradientBoostingClassifier<double> gb(
        Skigen::GradientBoostingClassifier<double>::Loss::LogLoss,
        0.1, /*n_estimators=*/30);
    gb.fit(X, y);
    Eigen::VectorXd s = gb.train_score();
    ASSERT_TRUE(s.size() == 30);
    ASSERT_TRUE(s(s.size() - 1) < s(0));
}

void test_gbc_classes_recorded() {
    constexpr int n = 20;
    Eigen::MatrixXd X(n, 1);
    Eigen::VectorXi y(n);
    for (int i = 0; i < n; ++i) {
        X(i, 0) = i;
        y(i)    = (i >= n / 2) ? 7 : 3;   // arbitrary class labels
    }
    Skigen::GradientBoostingClassifier<double> gb(
        Skigen::GradientBoostingClassifier<double>::Loss::LogLoss, 0.1, 20);
    gb.fit(X, y);
    ASSERT_TRUE(gb.classes()(0) == 3);
    ASSERT_TRUE(gb.classes()(1) == 7);
    ASSERT_TRUE(gb.n_classes() == 2);
}

void test_gbc_multiclass_throws() {
    Eigen::MatrixXd X(6, 1); X << 0, 1, 2, 3, 4, 5;
    Eigen::VectorXi y(6);    y << 0, 1, 2, 0, 1, 2;
    Skigen::GradientBoostingClassifier<double> gb(
        Skigen::GradientBoostingClassifier<double>::Loss::LogLoss, 0.1, 5);
    bool threw = false;
    try { gb.fit(X, y); }
    catch (const std::invalid_argument&) { threw = true; }
    ASSERT_TRUE(threw);
}

void test_gbc_exponential_loss_high_accuracy() {
    using GBC = Skigen::GradientBoostingClassifier<double>;
    constexpr int n = 200;
    Eigen::MatrixXd X(n, 2);
    Eigen::VectorXi y(n);
    std::mt19937_64 rng(13);
    std::normal_distribution<double> ns(0.0, 0.5);
    for (int i = 0; i < n; ++i) {
        const double cls = (i < n / 2) ? -1.0 : 1.0;
        X(i, 0) = cls + ns(rng);
        X(i, 1) = cls + ns(rng);
        y(i)    = (cls > 0) ? 1 : 0;
    }
    GBC gb(GBC::Loss::Exponential, 0.1, 100);
    gb.fit(X, y);
    auto preds = gb.predict(X);
    int correct = 0;
    for (int i = 0; i < n; ++i) if (preds(i) == y(i)) ++correct;
    ASSERT_TRUE(static_cast<double>(correct) / n > 0.95);

    const Eigen::MatrixXd proba = gb.predict_proba(X);
    for (int i = 0; i < n; ++i) {
        ASSERT_NEAR(proba(i, 0) + proba(i, 1), 1.0, 1e-12);
    }
    // Exponential train loss should decrease over the boosting iterations.
    ASSERT_TRUE(gb.train_score()(gb.train_score().size() - 1) <
                gb.train_score()(0));
}

void test_gbc_exponential_init_half_log_odds() {
    using GBC = Skigen::GradientBoostingClassifier<double>;
    Eigen::MatrixXd X(10, 1);
    Eigen::VectorXi y(10);
    for (int i = 0; i < 10; ++i) { X(i, 0) = i; y(i) = (i < 3) ? 1 : 0; }
    GBC gb(GBC::Loss::Exponential, 0.1, 5);
    gb.fit(X, y);
    const double p = 0.3;
    const double expected = 0.5 * std::log(p / (1.0 - p));
    ASSERT_NEAR(gb.init(), expected, 1e-9);
}

// ---------------------------------------------------------------------------
// HistGradientBoostingRegressor
// ---------------------------------------------------------------------------

void test_hgbr_recovers_linear_signal() {
    constexpr int n = 200;
    Eigen::MatrixXd X(n, 1);
    Eigen::VectorXd y(n);
    for (int i = 0; i < n; ++i) {
        const double x = -1.0 + 2.0 * static_cast<double>(i) / (n - 1);
        X(i, 0) = x;
        y(i)    = 3.0 * x + 1.0;
    }
    using HGBR = Skigen::HistGradientBoostingRegressor<double>;
    HGBR hgb(HGBR::Loss::SquaredError, /*lr=*/0.1,
             /*max_iter=*/100, /*max_leaf_nodes=*/31,
             /*max_depth=*/std::nullopt, /*min_samples_leaf=*/2);
    hgb.fit(X, y);
    ASSERT_TRUE(hgb.score(X, y) > 0.99);
}

void test_hgbr_init_equals_mean_of_y() {
    Eigen::MatrixXd X(5, 1); X << 0, 1, 2, 3, 4;
    Eigen::VectorXd y(5);    y << 10, 12, 11, 13, 14;
    Skigen::HistGradientBoostingRegressor<double> hgb;
    hgb.fit(X, y);
    ASSERT_NEAR(hgb.init(), y.mean(), 1e-12);
}

void test_hgbr_n_iter_honoured() {
    Eigen::MatrixXd X(20, 2);
    Eigen::VectorXd y(20);
    for (int i = 0; i < 20; ++i) {
        X(i, 0) = i; X(i, 1) = 20 - i;
        y(i) = 0.5 * X(i, 0) - 0.3 * X(i, 1);
    }
    using HGBR = Skigen::HistGradientBoostingRegressor<double>;
    HGBR hgb(HGBR::Loss::SquaredError, 0.1,
             /*max_iter=*/30, /*max_leaf_nodes=*/31,
             std::nullopt, 2);
    hgb.fit(X, y);
    ASSERT_TRUE(hgb.n_iter() == 30);
}

void test_hgbr_train_score_decreases() {
    Eigen::MatrixXd X(50, 1);
    Eigen::VectorXd y(50);
    for (int i = 0; i < 50; ++i) { X(i, 0) = i; y(i) = std::sin(0.2 * i); }
    using HGBR = Skigen::HistGradientBoostingRegressor<double>;
    HGBR hgb(HGBR::Loss::SquaredError, 0.1,
             /*max_iter=*/50, std::nullopt, std::nullopt, 2);
    hgb.fit(X, y);
    const Eigen::VectorXd s = hgb.train_score();
    ASSERT_TRUE(s.size() == 50);
    ASSERT_TRUE(s(s.size() - 1) < s(0));
}

void test_hgbr_bin_edges_per_feature() {
    Eigen::MatrixXd X(40, 3);
    for (int i = 0; i < 40; ++i) {
        X(i, 0) = static_cast<double>(i);          // many distinct values
        X(i, 1) = static_cast<double>(i % 4);      // few distinct values
        X(i, 2) = 0.0;                             // constant
    }
    Eigen::VectorXd y(40);
    for (int i = 0; i < 40; ++i) y(i) = X(i, 0);
    using HGBR = Skigen::HistGradientBoostingRegressor<double>;
    HGBR hgb(HGBR::Loss::SquaredError, 0.1,
             /*max_iter=*/10, std::nullopt, std::nullopt, 2,
             /*l2=*/0.0, /*max_bins=*/8);
    hgb.fit(X, y);
    const auto& edges = hgb.bin_edges();
    ASSERT_TRUE(edges.size() == 3);
    // Feature 0 has many distinct values → at most max_bins - 1 thresholds,
    // and we expect a healthy number of them.
    ASSERT_TRUE(edges[0].size() <= 7);
    ASSERT_TRUE(edges[0].size() >= 4);
    // Feature 1 has 4 distinct values → at most 4 unique thresholds.
    ASSERT_TRUE(edges[1].size() <= 4);
    // Feature 2 is constant → at most 1 unique threshold.
    ASSERT_TRUE(edges[2].size() <= 1);
}

void test_hgbr_unsupported_loss_throws() {
    using HGBR = Skigen::HistGradientBoostingRegressor<double>;
    bool threw = false;
    try {
        HGBR hgb(HGBR::Loss::Quantile);
        (void)hgb;
    } catch (const std::invalid_argument&) { threw = true; }
    ASSERT_TRUE(threw);
}

void test_hgbr_invalid_max_bins_throws() {
    using HGBR = Skigen::HistGradientBoostingRegressor<double>;
    bool threw = false;
    try {
        HGBR hgb(HGBR::Loss::SquaredError, 0.1, 100, 31, std::nullopt, 20,
                 0.0, /*max_bins=*/300);   // too large
        (void)hgb;
    } catch (const std::invalid_argument&) { threw = true; }
    ASSERT_TRUE(threw);
}

void test_hgbr_max_leaf_nodes_honoured() {
    // A small max_leaf_nodes should still fit a sensible model and keep the
    // per-tree leaf count bounded (leaf-wise growth).
    Eigen::MatrixXd X(120, 2);
    Eigen::VectorXd y(120);
    std::mt19937_64 rng(3);
    std::normal_distribution<double> ns(0.0, 0.05);
    for (int i = 0; i < 120; ++i) {
        X(i, 0) = static_cast<double>(i) / 120.0;
        X(i, 1) = std::sin(0.3 * i);
        y(i)    = 2.0 * X(i, 0) + X(i, 1) + ns(rng);
    }
    using HGBR = Skigen::HistGradientBoostingRegressor<double>;
    HGBR shallow(HGBR::Loss::SquaredError, 0.1, 60,
                 /*max_leaf_nodes=*/3, std::nullopt, 5, 0.0, 64,
                 std::nullopt, std::nullopt, false, 0.1, 10, 1e-7,
                 std::optional<uint64_t>(0));
    shallow.fit(X, y);
    ASSERT_TRUE(shallow.score(X, y) > 0.7);
    ASSERT_TRUE(shallow.n_iter() == 60);
}

void test_hgbr_l2_regularization_runs() {
    Eigen::MatrixXd X(80, 1);
    Eigen::VectorXd y(80);
    for (int i = 0; i < 80; ++i) { X(i, 0) = i; y(i) = 0.5 * i; }
    using HGBR = Skigen::HistGradientBoostingRegressor<double>;
    HGBR reg(HGBR::Loss::SquaredError, 0.1, 50, 31, std::nullopt, 5,
             /*l2=*/5.0, 32, std::nullopt, std::nullopt, false, 0.1, 10,
             1e-7, std::optional<uint64_t>(0));
    reg.fit(X, y);
    // L2 shrinks leaf values but the trend must still be learned.
    ASSERT_TRUE(reg.score(X, y) > 0.8);
}

void test_hgbr_monotonic_increasing() {
    // y is monotonically increasing in x; an increasing constraint must
    // produce non-decreasing predictions over sorted x.
    constexpr int n = 100;
    Eigen::MatrixXd X(n, 1);
    Eigen::VectorXd y(n);
    std::mt19937_64 rng(5);
    std::normal_distribution<double> ns(0.0, 0.3);
    for (int i = 0; i < n; ++i) {
        X(i, 0) = static_cast<double>(i) / 10.0;
        y(i)    = X(i, 0) + ns(rng);   // noisy but increasing
    }
    using HGBR = Skigen::HistGradientBoostingRegressor<double>;
    HGBR reg(HGBR::Loss::SquaredError, 0.1, 80, 15, std::nullopt, 5, 0.0,
             64, std::nullopt, std::vector<int>{+1}, false, 0.1, 10, 1e-7,
             std::optional<uint64_t>(0));
    reg.fit(X, y);
    // Evaluate on a sorted grid; predictions must be non-decreasing.
    Eigen::MatrixXd grid(50, 1);
    for (int i = 0; i < 50; ++i) grid(i, 0) = static_cast<double>(i) / 5.0;
    Eigen::VectorXd pred = reg.predict(grid);
    for (int i = 1; i < 50; ++i)
        ASSERT_TRUE(pred(i) >= pred(i - 1) - 1e-9);
}

void test_hgbr_native_categorical_split() {
    // A categorical feature whose target mapping is NON-monotonic in the
    // category code: cats {0, 2} -> +5, cats {1, 3} -> -5. An ordinal
    // (threshold) split cannot cleanly separate these, but the native
    // categorical (gradient-sorted subset) split can.
    constexpr int n = 160;
    Eigen::MatrixXd X(n, 1);
    Eigen::VectorXd y(n);
    std::mt19937_64 rng(4);
    std::normal_distribution<double> ns(0.0, 0.1);
    for (int i = 0; i < n; ++i) {
        const int cat = i % 4;
        X(i, 0) = static_cast<double>(cat);
        y(i) = ((cat == 0 || cat == 2) ? 5.0 : -5.0) + ns(rng);
    }
    using HGBR = Skigen::HistGradientBoostingRegressor<double>;
    // Mark feature 0 as categorical.
    HGBR cat(HGBR::Loss::SquaredError, 0.3, 60, 15, std::nullopt, 5, 0.0, 8,
             /*categorical_features=*/std::vector<int>{0},
             /*monotonic_cst=*/std::nullopt, false, 0.1, 10, 1e-7,
             std::optional<uint64_t>(0));
    cat.fit(X, y);
    // High R² requires resolving the non-monotonic category grouping.
    ASSERT_TRUE(cat.score(X, y) > 0.95);

    // The same data fit as ORDINAL cannot separate {0,2} from {1,3} well.
    HGBR ord(HGBR::Loss::SquaredError, 0.3, 60, 15, std::nullopt, 5, 0.0, 8,
             /*categorical_features=*/std::nullopt,
             /*monotonic_cst=*/std::nullopt, false, 0.1, 10, 1e-7,
             std::optional<uint64_t>(0));
    ord.fit(X, y);
    ASSERT_TRUE(cat.score(X, y) >= ord.score(X, y));
}

void test_hgbr_sparse_matches_dense() {
    // A mostly-zero design matrix; the native sparse HistGB path must match
    // the dense fit (same binning, same trees).
    constexpr int n = 120, p = 6;
    Eigen::MatrixXd Xd = Eigen::MatrixXd::Zero(n, p);
    Eigen::VectorXd y(n);
    std::mt19937_64 rng(8);
    std::uniform_int_distribution<int> col(0, p - 1);
    std::uniform_real_distribution<double> val(0.5, 3.0);
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < 2; ++k) Xd(i, col(rng)) = val(rng);
        y(i) = 2.0 * Xd(i, 0) - Xd(i, 1);
    }
    Eigen::SparseMatrix<double> Xs = Xd.sparseView();

    using HGBR = Skigen::HistGradientBoostingRegressor<double>;
    HGBR d(HGBR::Loss::SquaredError, 0.1, 40, 15, std::nullopt, 5, 0.0, 16,
           std::nullopt, std::nullopt, false, 0.1, 10, 1e-7,
           std::optional<uint64_t>(0));
    HGBR s(HGBR::Loss::SquaredError, 0.1, 40, 15, std::nullopt, 5, 0.0, 16,
           std::nullopt, std::nullopt, false, 0.1, 10, 1e-7,
           std::optional<uint64_t>(0));
    d.fit(Xd, y);
    s.fit(Xs, y);
    Eigen::VectorXd pd = d.predict(Xd);
    Eigen::VectorXd ps = s.predict(Xs);
    for (int i = 0; i < n; ++i) ASSERT_NEAR(pd(i), ps(i), 1e-10);
}

void test_hgbc_sparse_matches_dense() {
    constexpr int n = 120, p = 5;
    Eigen::MatrixXd Xd = Eigen::MatrixXd::Zero(n, p);
    Eigen::VectorXi y(n);
    std::mt19937_64 rng(9);
    std::uniform_int_distribution<int> col(0, p - 1);
    std::uniform_real_distribution<double> val(0.5, 3.0);
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < 2; ++k) Xd(i, col(rng)) = val(rng);
        y(i) = (Xd(i, 0) + Xd(i, 1) > 1.0) ? 1 : 0;
    }
    Eigen::SparseMatrix<double> Xs = Xd.sparseView();

    using HGBC = Skigen::HistGradientBoostingClassifier<double>;
    HGBC d(HGBC::Loss::LogLoss, 0.2, 40, 15, std::nullopt, 5, 0.0, 16,
           std::nullopt, std::nullopt, false, 0.1, 10, 1e-7,
           std::optional<uint64_t>(0));
    HGBC s(HGBC::Loss::LogLoss, 0.2, 40, 15, std::nullopt, 5, 0.0, 16,
           std::nullopt, std::nullopt, false, 0.1, 10, 1e-7,
           std::optional<uint64_t>(0));
    d.fit(Xd, y);
    s.fit(Xs, y);
    auto pd = d.predict(Xd);
    auto ps = s.predict(Xs);
    for (int i = 0; i < n; ++i) ASSERT_TRUE(pd(i) == ps(i));
}

void test_hgbc_native_categorical_split() {
    // Non-monotonic category->class mapping: cats {0,2} -> class 1,
    // {1,3} -> class 0. Requires a categorical (subset) split.
    constexpr int n = 200;
    Eigen::MatrixXd X(n, 1);
    Eigen::VectorXi y(n);
    for (int i = 0; i < n; ++i) {
        const int cat = i % 4;
        X(i, 0) = static_cast<double>(cat);
        y(i) = (cat == 0 || cat == 2) ? 1 : 0;
    }
    using HGBC = Skigen::HistGradientBoostingClassifier<double>;
    HGBC cat(HGBC::Loss::LogLoss, 0.3, 60, 15, std::nullopt, 5, 0.0, 8,
             /*monotonic_cst=*/std::nullopt,
             /*categorical_features=*/std::vector<int>{0}, false, 0.1, 10,
             1e-7, std::optional<uint64_t>(0));
    cat.fit(X, y);
    auto preds = cat.predict(X);
    int correct = 0;
    for (int i = 0; i < n; ++i) if (preds(i) == y(i)) ++correct;
    ASSERT_TRUE(static_cast<double>(correct) / n > 0.95);
}

void test_hgbr_early_stopping_truncates() {
    // With early stopping on a learnable signal, fitting should stop before
    // exhausting max_iter once the holdout stops improving.
    constexpr int n = 300;
    Eigen::MatrixXd X(n, 1);
    Eigen::VectorXd y(n);
    std::mt19937_64 rng(2);
    std::normal_distribution<double> ns(0.0, 0.05);
    for (int i = 0; i < n; ++i) {
        X(i, 0) = static_cast<double>(i) / n;
        y(i)    = std::sin(3.0 * X(i, 0)) + ns(rng);
    }
    using HGBR = Skigen::HistGradientBoostingRegressor<double>;
    HGBR reg(HGBR::Loss::SquaredError, 0.1, /*max_iter=*/500, 15,
             std::nullopt, 5, 0.0, 64, std::nullopt, std::nullopt,
             /*early_stopping=*/true, 0.2, /*n_iter_no_change=*/8, 1e-5,
             std::optional<uint64_t>(0));
    reg.fit(X, y);
    ASSERT_TRUE(reg.n_iter() < 500);
    ASSERT_TRUE(reg.score(X, y) > 0.8);
}

// ---------------------------------------------------------------------------
// HistGradientBoostingClassifier (binary, log-loss)
// ---------------------------------------------------------------------------

void test_hgbc_binary_separable_high_accuracy() {
    constexpr int n = 200;
    Eigen::MatrixXd X(n, 2);
    Eigen::VectorXi y(n);
    std::mt19937_64 rng(11);
    std::normal_distribution<double> ns(0.0, 0.5);
    for (int i = 0; i < n; ++i) {
        const double cls = (i < n / 2) ? -1.0 : 1.0;
        X(i, 0) = cls + ns(rng);
        X(i, 1) = cls + ns(rng);
        y(i)    = (cls > 0) ? 1 : 0;
    }
    using HGBC = Skigen::HistGradientBoostingClassifier<double>;
    HGBC hgb(HGBC::Loss::LogLoss, 0.1, 100, 31, std::nullopt, 2,
             0.0, 64, /*monotonic_cst=*/std::nullopt,
             /*categorical_features=*/std::nullopt, false, 0.1, 10, 1e-7,
             std::optional<uint64_t>(11));
    hgb.fit(X, y);
    auto preds = hgb.predict(X);
    int correct = 0;
    for (int i = 0; i < n; ++i) if (preds(i) == y(i)) ++correct;
    const double acc = static_cast<double>(correct) / n;
    ASSERT_TRUE(acc > 0.9);
}

void test_hgbc_predict_proba_rows_sum_to_one() {
    constexpr int n = 30;
    Eigen::MatrixXd X(n, 1);
    Eigen::VectorXi y(n);
    for (int i = 0; i < n; ++i) {
        X(i, 0) = i;
        y(i)    = (i >= n / 2) ? 1 : 0;
    }
    using HGBC = Skigen::HistGradientBoostingClassifier<double>;
    HGBC hgb(HGBC::Loss::LogLoss, 0.1, 50, 31, std::nullopt, 2);
    hgb.fit(X, y);
    Eigen::MatrixXd P = hgb.predict_proba(X);
    ASSERT_TRUE(P.rows() == n);
    ASSERT_TRUE(P.cols() == 2);
    for (int i = 0; i < n; ++i) {
        ASSERT_NEAR(P(i, 0) + P(i, 1), 1.0, 1e-12);
    }
}

void test_hgbc_init_log_odds() {
    constexpr int n = 50;
    Eigen::MatrixXd X(n, 1);
    Eigen::VectorXi y(n);
    for (int i = 0; i < n; ++i) {
        X(i, 0) = i;
        y(i)    = (i < n * 4 / 5) ? 1 : 0;          // 80/20 prior
    }
    Skigen::HistGradientBoostingClassifier<double> hgb;
    hgb.fit(X, y);
    // Binary init() is a length-1 vector of the prior log-odds.
    ASSERT_TRUE(hgb.init().size() == 1);
    ASSERT_NEAR(hgb.init()(0), std::log(4.0), 1e-9);
}

void test_hgbc_multiclass_separates_clusters() {
    constexpr int n = 150;
    Eigen::MatrixXd X(n, 2);
    Eigen::VectorXi y(n);
    std::mt19937_64 rng(7);
    std::normal_distribution<double> ns(0.0, 0.4);
    // Three clusters at the corners of a triangle.
    const double cx[3] = {0.0, 3.0, -3.0};
    const double cy[3] = {0.0, 3.0, 3.0};
    for (int i = 0; i < n; ++i) {
        const int k = i % 3;
        X(i, 0) = cx[k] + ns(rng);
        X(i, 1) = cy[k] + ns(rng);
        y(i)    = k;
    }
    Skigen::HistGradientBoostingClassifier<double> hgb(
        Skigen::HistGradientBoostingClassifier<double>::Loss::LogLoss,
        0.2, 80, 31, std::nullopt, 5, 0.0, 64, /*monotonic_cst=*/std::nullopt,
        /*categorical_features=*/std::nullopt, false, 0.1, 10, 1e-7,
        std::optional<uint64_t>(0));
    hgb.fit(X, y);
    ASSERT_TRUE(hgb.n_classes() == 3);
    Eigen::MatrixXd P = hgb.predict_proba(X);
    ASSERT_TRUE(P.cols() == 3);
    for (int i = 0; i < n; ++i)
        ASSERT_NEAR(P.row(i).sum(), 1.0, 1e-10);
    auto preds = hgb.predict(X);
    int correct = 0;
    for (int i = 0; i < n; ++i) if (preds(i) == y(i)) ++correct;
    ASSERT_TRUE(static_cast<double>(correct) / n > 0.9);
}

// ---------------------------------------------------------------------------

int main() {
    std::cout << "=== RandomForestClassifier Tests ===\n";
    run_test("rfc_two_class_linear",            test_rfc_two_class_linear);
    run_test("rfc_three_class_xor",             test_rfc_three_class_xor);
    run_test("rfc_predict_proba",               test_rfc_predict_proba);
    run_test("rfc_n_estimators_honoured",       test_rfc_n_estimators_honoured);
    run_test("rfc_feature_importances_normalised", test_rfc_feature_importances_normalised);
    run_test("rfc_oob_score_populated",         test_rfc_oob_score_populated);
    run_test("rfc_random_state_reproducible",   test_rfc_random_state_reproducible);
    run_test("rfc_bootstrap_false",             test_rfc_bootstrap_false);
    run_test("rfc_n_jobs_equivalence",          test_rfc_n_jobs_equivalence);
    run_test("rfc_unsupported_criterion_throws", test_rfc_unsupported_criterion_throws);
    run_test("rfc_max_features_modes",          test_rfc_max_features_modes);
    run_test("rfc_feature_count_mismatch_throws", test_rfc_feature_count_mismatch_throws);
    run_test("rfc_not_fitted_throws",           test_rfc_not_fitted_throws);

    std::cout << "\n=== RandomForestRegressor Tests ===\n";
    run_test("rfr_noisy_linear",                test_rfr_noisy_linear);
    run_test("rfr_predict_matches_mean_of_two_trees", test_rfr_predict_matches_mean_of_two_trees);
    run_test("rfr_unsupported_criterion_throws", test_rfr_unsupported_criterion_throws);
    run_test("rfr_multi_output_predicts_two_targets", test_rfr_multi_output_predicts_two_targets);

    std::cout << "\n=== GradientBoostingRegressor Tests ===\n";
    run_test("gbr_recovers_linear_signal",          test_gbr_recovers_linear_signal);
    run_test("gbr_init_equals_mean_of_y",           test_gbr_init_equals_mean_of_y);
    run_test("gbr_n_estimators_honoured",           test_gbr_n_estimators_honoured);
    run_test("gbr_train_score_decreases",           test_gbr_train_score_decreases);
    run_test("gbr_feature_importances_shape_and_normalised",
             test_gbr_feature_importances_shape_and_normalised);
    run_test("gbr_absolute_error_recovers_signal",  test_gbr_absolute_error_recovers_signal);
    run_test("gbr_quantile_loss_tracks_quantile",   test_gbr_quantile_loss_tracks_quantile);
    run_test("gbr_quantile_invalid_alpha_throws",   test_gbr_quantile_invalid_alpha_throws);
    run_test("gbr_stochastic_subsample_fits_and_is_deterministic",
             test_gbr_stochastic_subsample_fits_and_is_deterministic);
    run_test("gbr_invalid_subsample_throws",        test_gbr_invalid_subsample_throws);
    run_test("gbr_not_fitted_throws",               test_gbr_not_fitted_throws);

    std::cout << "\n=== GradientBoostingClassifier Tests ===\n";
    run_test("gbc_binary_separable_high_accuracy",            test_gbc_binary_separable_high_accuracy);
    run_test("gbc_predict_proba_shape_and_rows_sum_to_one",   test_gbc_predict_proba_shape_and_rows_sum_to_one);
    run_test("gbc_init_log_odds",                             test_gbc_init_log_odds);
    run_test("gbc_train_score_decreases",                     test_gbc_train_score_decreases);
    run_test("gbc_classes_recorded",                          test_gbc_classes_recorded);
    run_test("gbc_multiclass_throws",                         test_gbc_multiclass_throws);
    run_test("gbc_exponential_loss_high_accuracy",            test_gbc_exponential_loss_high_accuracy);
    run_test("gbc_exponential_init_half_log_odds",            test_gbc_exponential_init_half_log_odds);

    std::cout << "\n=== HistGradientBoostingRegressor Tests ===\n";
    run_test("hgbr_recovers_linear_signal",       test_hgbr_recovers_linear_signal);
    run_test("hgbr_init_equals_mean_of_y",        test_hgbr_init_equals_mean_of_y);
    run_test("hgbr_n_iter_honoured",              test_hgbr_n_iter_honoured);
    run_test("hgbr_train_score_decreases",        test_hgbr_train_score_decreases);
    run_test("hgbr_bin_edges_per_feature",        test_hgbr_bin_edges_per_feature);
    run_test("hgbr_unsupported_loss_throws",      test_hgbr_unsupported_loss_throws);
    run_test("hgbr_invalid_max_bins_throws",      test_hgbr_invalid_max_bins_throws);
    run_test("hgbr_max_leaf_nodes_honoured",      test_hgbr_max_leaf_nodes_honoured);
    run_test("hgbr_l2_regularization_runs",       test_hgbr_l2_regularization_runs);
    run_test("hgbr_monotonic_increasing",         test_hgbr_monotonic_increasing);
    run_test("hgbr_native_categorical_split",     test_hgbr_native_categorical_split);
    run_test("hgbr_sparse_matches_dense",          test_hgbr_sparse_matches_dense);
    run_test("hgbr_early_stopping_truncates",      test_hgbr_early_stopping_truncates);

    std::cout << "\n=== HistGradientBoostingClassifier Tests ===\n";
    run_test("hgbc_binary_separable_high_accuracy",
             test_hgbc_binary_separable_high_accuracy);
    run_test("hgbc_predict_proba_rows_sum_to_one",
             test_hgbc_predict_proba_rows_sum_to_one);
    run_test("hgbc_init_log_odds",                test_hgbc_init_log_odds);
    run_test("hgbc_multiclass_separates_clusters",
             test_hgbc_multiclass_separates_clusters);
    run_test("hgbc_native_categorical_split",     test_hgbc_native_categorical_split);
    run_test("hgbc_sparse_matches_dense",         test_hgbc_sparse_matches_dense);

    std::cout << "\n" << g_passed << " passed, " << g_failed << " failed.\n";
    return g_failed > 0 ? 1 : 0;
}
