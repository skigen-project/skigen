// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#include <Skigen/Dense>

#include <cmath>
#include <functional>
#include <iostream>
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

    std::cout << "\n" << g_passed << " passed, " << g_failed << " failed.\n";
    return g_failed > 0 ? 1 : 0;
}
