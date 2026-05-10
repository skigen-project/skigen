// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#include <Skigen/Isotonic>

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <functional>
#include <vector>

// ---------------------------------------------------------------------------
// Minimal test harness (matches sibling test files)
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
    } while (0)

#define ASSERT_NEAR(a, b, tol)                                                 \
    do {                                                                       \
        const double aa = static_cast<double>(a);                              \
        const double bb = static_cast<double>(b);                              \
        if (std::fabs(aa - bb) > (tol)) {                                      \
            std::ostringstream oss;                                            \
            oss << __FILE__ << ":" << __LINE__                                 \
                << ": ASSERT_NEAR failed: " << aa << " vs " << bb              \
                << " (tol=" << (tol) << ")";                                   \
            throw TestFailure(oss.str());                                      \
        }                                                                      \
    } while (0)

#define ASSERT_THROWS(stmt, ExType)                                            \
    do {                                                                       \
        bool caught = false;                                                   \
        try { stmt; } catch (const ExType&) { caught = true; }                 \
        if (!caught) {                                                         \
            throw TestFailure(std::string(__FILE__) + ":" +                    \
                              std::to_string(__LINE__) +                       \
                              ": expected exception " #ExType " not thrown");  \
        }                                                                      \
    } while (0)

static void run(const std::string& name, std::function<void()> body) {
    try {
        body();
        ++g_passed;
        std::cout << "  [PASS] " << name << "\n";
    } catch (const TestFailure& e) {
        ++g_failed;
        std::cout << "  [FAIL] " << name << " — " << e.what() << "\n";
    } catch (const std::exception& e) {
        ++g_failed;
        std::cout << "  [FAIL] " << name << " — unexpected: " << e.what() << "\n";
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

using Skigen::IsotonicRegression;
using Skigen::IsotonicIncreasing;
using Skigen::OutOfBounds;

void test_iso_already_increasing() {
    Eigen::MatrixXd X(5, 1);
    X << 1, 2, 3, 4, 5;
    Eigen::VectorXd y(5);
    y << 1, 2, 3, 4, 5;

    IsotonicRegression<double> iso;
    iso.fit(X, y);
    Eigen::VectorXd yhat = iso.predict(X);
    for (int i = 0; i < 5; ++i) ASSERT_NEAR(yhat(i), y(i), 1e-12);
}

void test_iso_violators_pooled() {
    // Classic PAVA example: y = [1, 3, 2, 4] -> pool the 3,2 to 2.5.
    Eigen::MatrixXd X(4, 1);
    X << 1, 2, 3, 4;
    Eigen::VectorXd y(4);
    y << 1, 3, 2, 4;

    IsotonicRegression<double> iso;
    iso.fit(X, y);
    Eigen::VectorXd yhat = iso.predict(X);
    ASSERT_NEAR(yhat(0), 1.0, 1e-12);
    ASSERT_NEAR(yhat(1), 2.5, 1e-12);
    ASSERT_NEAR(yhat(2), 2.5, 1e-12);
    ASSERT_NEAR(yhat(3), 4.0, 1e-12);
}

void test_iso_monotone_output() {
    // After fitting, yhat must be non-decreasing along sorted X.
    Eigen::MatrixXd X(8, 1);
    X << 1, 2, 3, 4, 5, 6, 7, 8;
    Eigen::VectorXd y(8);
    y << 5, 1, 4, 2, 6, 3, 7, 8;

    IsotonicRegression<double> iso;
    iso.fit(X, y);
    Eigen::VectorXd yhat = iso.predict(X);
    for (int i = 1; i < 8; ++i) {
        ASSERT_TRUE(yhat(i) >= yhat(i - 1) - 1e-12);
    }
}

void test_iso_decreasing() {
    Eigen::MatrixXd X(5, 1);
    X << 1, 2, 3, 4, 5;
    Eigen::VectorXd y(5);
    y << 5, 4, 3, 2, 1;

    IsotonicRegression<double> iso(std::nullopt, std::nullopt,
                                   IsotonicIncreasing::False);
    iso.fit(X, y);
    Eigen::VectorXd yhat = iso.predict(X);
    for (int i = 1; i < 5; ++i) {
        ASSERT_TRUE(yhat(i) <= yhat(i - 1) + 1e-12);
    }
}

void test_iso_auto_picks_decreasing() {
    Eigen::MatrixXd X(5, 1);
    X << 1, 2, 3, 4, 5;
    Eigen::VectorXd y(5);
    y << 10, 8, 6, 4, 2;

    IsotonicRegression<double> iso(std::nullopt, std::nullopt,
                                   IsotonicIncreasing::Auto);
    iso.fit(X, y);
    ASSERT_TRUE(iso.increasing_resolved() == false);
    Eigen::VectorXd yhat = iso.predict(X);
    for (int i = 1; i < 5; ++i) {
        ASSERT_TRUE(yhat(i) <= yhat(i - 1) + 1e-12);
    }
}

void test_iso_y_bounds() {
    Eigen::MatrixXd X(4, 1);
    X << 1, 2, 3, 4;
    Eigen::VectorXd y(4);
    y << 0, 5, 10, 20;

    IsotonicRegression<double> iso(/*y_min=*/2.0, /*y_max=*/15.0);
    iso.fit(X, y);
    Eigen::VectorXd yhat = iso.predict(X);
    for (int i = 0; i < 4; ++i) {
        ASSERT_TRUE(yhat(i) >= 2.0 - 1e-12);
        ASSERT_TRUE(yhat(i) <= 15.0 + 1e-12);
    }
}

void test_iso_interpolate_between_thresholds() {
    Eigen::MatrixXd X(4, 1);
    X << 0, 1, 2, 3;
    Eigen::VectorXd y(4);
    y << 0, 1, 2, 3;
    IsotonicRegression<double> iso;
    iso.fit(X, y);

    Eigen::MatrixXd T(3, 1);
    T << 0.5, 1.5, 2.5;
    Eigen::VectorXd yhat = iso.predict(T);
    ASSERT_NEAR(yhat(0), 0.5, 1e-12);
    ASSERT_NEAR(yhat(1), 1.5, 1e-12);
    ASSERT_NEAR(yhat(2), 2.5, 1e-12);
}

void test_iso_out_of_bounds_nan() {
    Eigen::MatrixXd X(3, 1); X << 1, 2, 3;
    Eigen::VectorXd y(3); y << 1, 2, 3;
    IsotonicRegression<double> iso;  // default Nan
    iso.fit(X, y);
    Eigen::MatrixXd T(2, 1); T << 0, 4;
    Eigen::VectorXd out = iso.predict(T);
    ASSERT_TRUE(std::isnan(out(0)));
    ASSERT_TRUE(std::isnan(out(1)));
}

void test_iso_out_of_bounds_clip() {
    Eigen::MatrixXd X(3, 1); X << 1, 2, 3;
    Eigen::VectorXd y(3); y << 1, 2, 3;
    IsotonicRegression<double> iso(std::nullopt, std::nullopt,
                                   IsotonicIncreasing::True,
                                   OutOfBounds::Clip);
    iso.fit(X, y);
    Eigen::MatrixXd T(2, 1); T << -5, 100;
    Eigen::VectorXd out = iso.predict(T);
    ASSERT_NEAR(out(0), 1.0, 1e-12);
    ASSERT_NEAR(out(1), 3.0, 1e-12);
}

void test_iso_out_of_bounds_raise() {
    Eigen::MatrixXd X(3, 1); X << 1, 2, 3;
    Eigen::VectorXd y(3); y << 1, 2, 3;
    IsotonicRegression<double> iso(std::nullopt, std::nullopt,
                                   IsotonicIncreasing::True,
                                   OutOfBounds::Raise);
    iso.fit(X, y);
    Eigen::MatrixXd T(1, 1); T << 100;
    ASSERT_THROWS((void)iso.predict(T), std::invalid_argument);
}

void test_iso_transform_alias() {
    Eigen::MatrixXd X(4, 1); X << 1, 2, 3, 4;
    Eigen::VectorXd y(4); y << 1, 3, 2, 4;
    IsotonicRegression<double> iso;
    iso.fit(X, y);
    Eigen::VectorXd p = iso.predict(X);
    Eigen::VectorXd t = iso.transform(X);
    for (int i = 0; i < 4; ++i) ASSERT_NEAR(p(i), t(i), 1e-12);
}

void test_iso_score_perfect() {
    Eigen::MatrixXd X(5, 1); X << 1, 2, 3, 4, 5;
    Eigen::VectorXd y(5); y << 1, 2, 3, 4, 5;
    IsotonicRegression<double> iso;
    iso.fit(X, y);
    ASSERT_NEAR(iso.score(X, y), 1.0, 1e-12);
}

void test_iso_multi_column_throws() {
    Eigen::MatrixXd X(3, 2); X.setOnes();
    Eigen::VectorXd y(3); y << 1, 2, 3;
    IsotonicRegression<double> iso;
    ASSERT_THROWS(iso.fit(X, y), std::invalid_argument);
}

void test_iso_not_fitted() {
    IsotonicRegression<double> iso;
    Eigen::MatrixXd X(2, 1); X << 1, 2;
    ASSERT_THROWS((void)iso.predict(X), std::runtime_error);
}

void test_iso_ties_on_x() {
    // Repeated X values get merged into a single block before PAVA.
    Eigen::MatrixXd X(6, 1); X << 1, 1, 2, 2, 3, 3;
    Eigen::VectorXd y(6); y << 0, 2, 5, 1, 4, 4;
    IsotonicRegression<double> iso;
    iso.fit(X, y);
    Eigen::MatrixXd Q(3, 1); Q << 1, 2, 3;
    Eigen::VectorXd q = iso.predict(Q);
    // Block means: x=1 -> 1; x=2 -> 3; x=3 -> 4 — already monotone.
    ASSERT_NEAR(q(0), 1.0, 1e-12);
    ASSERT_NEAR(q(1), 3.0, 1e-12);
    ASSERT_NEAR(q(2), 4.0, 1e-12);
}

int main() {
    std::cout << "Skigen IsotonicRegression unit tests\n";
    std::cout << "------------------------------------\n";
    run("already_increasing",         test_iso_already_increasing);
    run("violators_pooled",           test_iso_violators_pooled);
    run("monotone_output",            test_iso_monotone_output);
    run("decreasing",                 test_iso_decreasing);
    run("auto_picks_decreasing",      test_iso_auto_picks_decreasing);
    run("y_bounds",                   test_iso_y_bounds);
    run("interpolate_between_thresholds", test_iso_interpolate_between_thresholds);
    run("out_of_bounds_nan",          test_iso_out_of_bounds_nan);
    run("out_of_bounds_clip",         test_iso_out_of_bounds_clip);
    run("out_of_bounds_raise",        test_iso_out_of_bounds_raise);
    run("transform_alias",            test_iso_transform_alias);
    run("score_perfect",              test_iso_score_perfect);
    run("multi_column_throws",        test_iso_multi_column_throws);
    run("not_fitted",                 test_iso_not_fitted);
    run("ties_on_x",                  test_iso_ties_on_x);

    std::cout << "------------------------------------\n";
    std::cout << g_passed << " passed, " << g_failed << " failed\n";
    return g_failed == 0 ? 0 : 1;
}
