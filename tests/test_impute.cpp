// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

#include <Skigen/Dense>

#include <Eigen/Core>
#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

static int g_passed = 0;
static int g_failed = 0;

struct TestFailure : std::exception {
    std::string msg;
    explicit TestFailure(std::string m) : msg(std::move(m)) {}
    const char* what() const noexcept override { return msg.c_str(); }
};

#define ASSERT_TRUE(cond)                                                     \
    do {                                                                      \
        if (!(cond)) {                                                        \
            throw TestFailure(std::string(__FILE__) + ":" +                  \
                              std::to_string(__LINE__) + ": ASSERT_TRUE(" +  \
                              #cond + ") failed");                           \
        }                                                                     \
    } while (false)

#define ASSERT_THROW(expr, ExType)                                             \
    do {                                                                       \
        bool caught = false;                                                   \
        try {                                                                  \
            static_cast<void>(expr);                                           \
        } catch (const ExType&) {                                              \
            caught = true;                                                     \
        }                                                                      \
        if (!caught) {                                                         \
            throw TestFailure(std::string(__FILE__) + ":" +                   \
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

template <typename MatA, typename MatB, typename Scalar>
void assert_matrix_near(const MatA& A, const MatB& B, Scalar tol,
                        const char* file, int line) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        std::ostringstream oss;
        oss << file << ":" << line << ": Shape mismatch: got "
            << A.rows() << "x" << A.cols() << " expected "
            << B.rows() << "x" << B.cols();
        throw TestFailure(oss.str());
    }
    for (Eigen::Index i = 0; i < A.rows(); ++i) {
        for (Eigen::Index j = 0; j < A.cols(); ++j) {
            assert_near(A(i, j), B(i, j), tol, file, line);
        }
    }
}
#define ASSERT_MATRIX_NEAR(A, B, tol) assert_matrix_near(A, B, tol, __FILE__, __LINE__)

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
        std::cout << "  FAIL  " << name << "\n        exception: " << e.what() << "\n";
    }
}

void test_simple_imputer_mean_nan() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    Eigen::MatrixXd X(3, 3);
    X << 1.0, nan, 3.0,
         2.0, 4.0, nan,
         nan, 6.0, 9.0;

    Skigen::SimpleImputer<double> imputer;
    imputer.fit(X);
    Eigen::MatrixXd Z = imputer.transform(X);

    Eigen::RowVector3d expected_stats;
    expected_stats << 1.5, 5.0, 6.0;
    Eigen::MatrixXd expected(3, 3);
    expected << 1.0, 5.0, 3.0,
                2.0, 4.0, 6.0,
                1.5, 6.0, 9.0;

    ASSERT_MATRIX_NEAR(imputer.statistics(), expected_stats, 1e-12);
    ASSERT_MATRIX_NEAR(Z, expected, 1e-12);
}

void test_simple_imputer_median_marker() {
    Eigen::MatrixXd X(4, 2);
    X << 1.0, 2.0,
        -1.0, 4.0,
         5.0, -1.0,
         7.0, 8.0;

    Skigen::SimpleImputer<double> imputer(-1.0, Skigen::ImputeStrategy::Median);
    imputer.fit(X);

    Eigen::RowVector2d expected_stats;
    expected_stats << 5.0, 4.0;
    ASSERT_MATRIX_NEAR(imputer.statistics(), expected_stats, 1e-12);
    ASSERT_NEAR(imputer.transform(X)(1, 0), 5.0, 1e-12);
    ASSERT_NEAR(imputer.transform(X)(2, 1), 4.0, 1e-12);
}

void test_simple_imputer_most_frequent_tie() {
    Eigen::MatrixXd X(5, 1);
    X << 2.0, 1.0, 2.0, 1.0, -1.0;

    Skigen::SimpleImputer<double> imputer(
        -1.0, Skigen::ImputeStrategy::MostFrequent);
    imputer.fit(X);

    ASSERT_NEAR(imputer.statistics()(0), 1.0, 1e-12);
    ASSERT_NEAR(imputer.transform(X)(4, 0), 1.0, 1e-12);
}

void test_simple_imputer_constant_add_indicator() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    Eigen::MatrixXd X(3, 3);
    X << nan, 2.0, 3.0,
         4.0, 5.0, nan,
         7.0, 8.0, 9.0;

    Skigen::SimpleImputer<double> imputer(
        nan, Skigen::ImputeStrategy::Constant, -5.0,
        /*add_indicator=*/true);
    Eigen::MatrixXd Z = imputer.fit_transform(X);

    Eigen::MatrixXd expected(3, 5);
    expected << -5.0, 2.0, 3.0, 1.0, 0.0,
                 4.0, 5.0, -5.0, 0.0, 1.0,
                 7.0, 8.0, 9.0, 0.0, 0.0;
    ASSERT_MATRIX_NEAR(Z, expected, 1e-12);
    ASSERT_TRUE(imputer.indicator_features().size() == 2);
    ASSERT_TRUE(imputer.indicator_features()(0) == 0);
    ASSERT_TRUE(imputer.indicator_features()(1) == 2);
}

void test_simple_imputer_empty_feature_drop_and_keep() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    Eigen::MatrixXd X(2, 2);
    X << nan, 1.0,
         nan, 2.0;

    Skigen::SimpleImputer<double> dropper;
    Eigen::MatrixXd dropped = dropper.fit_transform(X);
    ASSERT_TRUE(dropped.rows() == 2);
    ASSERT_TRUE(dropped.cols() == 1);
    ASSERT_NEAR(dropped(0, 0), 1.0, 1e-12);
    ASSERT_TRUE(std::isnan(dropper.statistics()(0)));

    Skigen::SimpleImputer<double> keeper(
        nan, Skigen::ImputeStrategy::Mean, 0.0,
        /*add_indicator=*/false,
        /*keep_empty_features=*/true);
    Eigen::MatrixXd kept = keeper.fit_transform(X);
    Eigen::MatrixXd expected(2, 2);
    expected << 0.0, 1.0,
                0.0, 2.0;
    ASSERT_MATRIX_NEAR(kept, expected, 1e-12);
}

void test_missing_indicator_missing_only_error_on_new() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    Eigen::MatrixXd X(2, 2);
    X << nan, 1.0,
         2.0, 3.0;

    Skigen::MissingIndicator<double> indicator;
    indicator.fit(X);
    ASSERT_TRUE(indicator.features_indices().size() == 1);
    ASSERT_TRUE(indicator.features_indices()(0) == 0);

    Eigen::MatrixXd X_new(1, 2);
    X_new << 4.0, nan;
    ASSERT_THROW(indicator.transform(X_new), std::invalid_argument);
}

void test_missing_indicator_all_and_no_new_error() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    Eigen::MatrixXd X(2, 2);
    X << nan, 1.0,
         2.0, 3.0;

    Skigen::MissingIndicator<double> all(
        nan, Skigen::MissingIndicatorFeatures::All);
    Eigen::MatrixXd mask = all.fit_transform(X);
    Eigen::MatrixXd expected(2, 2);
    expected << 1.0, 0.0,
                0.0, 0.0;
    ASSERT_MATRIX_NEAR(mask, expected, 1e-12);

    Skigen::MissingIndicator<double> missing_only(
        nan, Skigen::MissingIndicatorFeatures::MissingOnly,
        /*error_on_new=*/false);
    missing_only.fit(X);
    Eigen::MatrixXd X_new(1, 2);
    X_new << 4.0, nan;
    Eigen::MatrixXd new_mask = missing_only.transform(X_new);
    ASSERT_TRUE(new_mask.rows() == 1);
    ASSERT_TRUE(new_mask.cols() == 1);
    ASSERT_NEAR(new_mask(0, 0), 0.0, 1e-12);
}

int main() {
    std::cout << "Running Impute tests...\n";
    run_test("SimpleImputer mean with NaN", test_simple_imputer_mean_nan);
    run_test("SimpleImputer median with marker", test_simple_imputer_median_marker);
    run_test("SimpleImputer most frequent tie", test_simple_imputer_most_frequent_tie);
    run_test("SimpleImputer constant + indicator", test_simple_imputer_constant_add_indicator);
    run_test("SimpleImputer empty feature drop/keep", test_simple_imputer_empty_feature_drop_and_keep);
    run_test("MissingIndicator new missing error", test_missing_indicator_missing_only_error_on_new);
    run_test("MissingIndicator all/no-new-error", test_missing_indicator_all_and_no_new_error);

    std::cout << "\nPassed: " << g_passed << ", Failed: " << g_failed << "\n";
    return g_failed == 0 ? 0 : 1;
}
