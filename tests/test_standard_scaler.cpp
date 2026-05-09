// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#include <Skigen/Dense>

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <functional>
#include <vector>

// ---------------------------------------------------------------------------
// Minimal test harness (no external dependency)
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
            << " vs " << b << " (tol=" << tol << ", diff=" << std::abs(a - b)
            << ")";
        throw TestFailure(oss.str());
    }
}

#define ASSERT_NEAR(a, b, tol) assert_near(a, b, tol, __FILE__, __LINE__)

template <typename MatA, typename MatB, typename Scalar>
void assert_matrix_near(const MatA& A, const MatB& B, Scalar tol,
                        const char* file, int line) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        std::ostringstream oss;
        oss << file << ":" << line << ": Shape mismatch: (" << A.rows() << ","
            << A.cols() << ") vs (" << B.rows() << "," << B.cols() << ")";
        throw TestFailure(oss.str());
    }
    for (Eigen::Index i = 0; i < A.rows(); ++i) {
        for (Eigen::Index j = 0; j < A.cols(); ++j) {
            assert_near(A(i, j), B(i, j), tol, file, line);
        }
    }
}

#define ASSERT_MATRIX_NEAR(A, B, tol)                                         \
    assert_matrix_near(A, B, tol, __FILE__, __LINE__)

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
// Tests
// ---------------------------------------------------------------------------

void test_fit_known_data() {
    // X = [[1, 2], [3, 4], [5, 6]]
    // mean = [3, 4], var = [8/3, 8/3], scale = sqrt(8/3)
    Eigen::MatrixXd X(3, 2);
    X << 1, 2,
         3, 4,
         5, 6;

    Skigen::StandardScaler scaler;
    scaler.fit(X);

    ASSERT_TRUE(scaler.is_fitted());
    ASSERT_NEAR(scaler.mean()(0), 3.0, 1e-12);
    ASSERT_NEAR(scaler.mean()(1), 4.0, 1e-12);
    ASSERT_NEAR(scaler.var()(0), 8.0 / 3.0, 1e-12);
    ASSERT_NEAR(scaler.var()(1), 8.0 / 3.0, 1e-12);
    ASSERT_NEAR(scaler.scale()(0), std::sqrt(8.0 / 3.0), 1e-12);
    ASSERT_NEAR(scaler.scale()(1), std::sqrt(8.0 / 3.0), 1e-12);
    ASSERT_TRUE(scaler.n_samples_seen() == 3);
    ASSERT_TRUE(scaler.n_features_in() == 2);
}

void test_transform_correct_zscores() {
    Eigen::MatrixXd X(3, 2);
    X << 1, 2,
         3, 4,
         5, 6;

    Skigen::StandardScaler scaler;
    scaler.fit(X);
    Eigen::MatrixXd Z = scaler.transform(X);

    double s = std::sqrt(8.0 / 3.0);
    ASSERT_NEAR(Z(0, 0), (1.0 - 3.0) / s, 1e-12);
    ASSERT_NEAR(Z(1, 0), (3.0 - 3.0) / s, 1e-12);
    ASSERT_NEAR(Z(2, 0), (5.0 - 3.0) / s, 1e-12);
    ASSERT_NEAR(Z(0, 1), (2.0 - 4.0) / s, 1e-12);
    ASSERT_NEAR(Z(1, 1), (4.0 - 4.0) / s, 1e-12);
    ASSERT_NEAR(Z(2, 1), (6.0 - 4.0) / s, 1e-12);
}

void test_round_trip() {
    Eigen::MatrixXd X(4, 3);
    X << 1, 10, 100,
         2, 20, 200,
         3, 30, 300,
         4, 40, 400;

    Skigen::StandardScaler scaler;
    scaler.fit(X);
    Eigen::MatrixXd Z = scaler.transform(X);
    Eigen::MatrixXd X_back = scaler.inverse_transform(Z);

    ASSERT_MATRIX_NEAR(X, X_back, 1e-12);
}

void test_fit_transform_equivalence() {
    Eigen::MatrixXd X(3, 2);
    X << 1, 2,
         3, 4,
         5, 6;

    Skigen::StandardScaler s1;
    s1.fit(X);
    Eigen::MatrixXd Z1 = s1.transform(X);

    Skigen::StandardScaler s2;
    Eigen::MatrixXd Z2 = s2.fit_transform(X);

    ASSERT_MATRIX_NEAR(Z1, Z2, 1e-15);
}

void test_near_zero_variance() {
    // Column 1 is constant → variance ≈ 0 → scale should be 1.0
    Eigen::MatrixXd X(3, 2);
    X << 1, 5,
         2, 5,
         3, 5;

    Skigen::StandardScaler scaler;
    scaler.fit(X);

    ASSERT_NEAR(scaler.scale()(1), 1.0, 1e-12);
}

void test_without_mean() {
    Eigen::MatrixXd X(3, 2);
    X << 1, 2,
         3, 4,
         5, 6;

    Skigen::StandardScaler scaler(false, true);
    scaler.fit(X);
    Eigen::MatrixXd Z = scaler.transform(X);

    // Should only scale, not center
    double s = std::sqrt(8.0 / 3.0);
    ASSERT_NEAR(Z(0, 0), 1.0 / s, 1e-12);
    ASSERT_NEAR(Z(1, 0), 3.0 / s, 1e-12);
}

void test_without_std() {
    Eigen::MatrixXd X(3, 2);
    X << 1, 2,
         3, 4,
         5, 6;

    Skigen::StandardScaler scaler(true, false);
    scaler.fit(X);
    Eigen::MatrixXd Z = scaler.transform(X);

    // Should only center, not scale
    ASSERT_NEAR(Z(0, 0), 1.0 - 3.0, 1e-12);
    ASSERT_NEAR(Z(1, 0), 3.0 - 3.0, 1e-12);
    ASSERT_NEAR(Z(2, 0), 5.0 - 3.0, 1e-12);
}

void test_transform_inplace() {
    Eigen::MatrixXd X(3, 2);
    X << 1, 2,
         3, 4,
         5, 6;

    Skigen::StandardScaler scaler;
    scaler.fit(X);

    Eigen::MatrixXd Z_copy = scaler.transform(X);

    Eigen::MatrixXd X_mut = X;
    scaler.transform_inplace(X_mut);

    ASSERT_MATRIX_NEAR(Z_copy, X_mut, 1e-15);
}

void test_single_feature() {
    Eigen::MatrixXd X(4, 1);
    X << 2, 4, 6, 8;

    Skigen::StandardScaler scaler;
    scaler.fit(X);
    Eigen::MatrixXd Z = scaler.transform(X);

    ASSERT_NEAR(scaler.mean()(0), 5.0, 1e-12);
    // var = ((2-5)^2 + (4-5)^2 + (6-5)^2 + (8-5)^2) / 4 = (9+1+1+9)/4 = 5
    ASSERT_NEAR(scaler.var()(0), 5.0, 1e-12);
}

void test_single_sample() {
    Eigen::MatrixXd X(1, 3);
    X << 10, 20, 30;

    Skigen::StandardScaler scaler;
    scaler.fit(X);

    // Variance is 0 for single sample → scale should be 1.0
    ASSERT_NEAR(scaler.scale()(0), 1.0, 1e-12);
    ASSERT_NEAR(scaler.scale()(1), 1.0, 1e-12);
    ASSERT_NEAR(scaler.scale()(2), 1.0, 1e-12);
}

void test_feature_mismatch_throws() {
    Eigen::MatrixXd X(3, 2);
    X << 1, 2, 3, 4, 5, 6;

    Skigen::StandardScaler scaler;
    scaler.fit(X);

    Eigen::MatrixXd X_bad(3, 3);
    X_bad << 1, 2, 3, 4, 5, 6, 7, 8, 9;

    ASSERT_THROW(scaler.transform(X_bad), std::invalid_argument);
}

void test_not_fitted_throws() {
    Skigen::StandardScaler scaler;
    Eigen::MatrixXd X(2, 2);
    X << 1, 2, 3, 4;

    ASSERT_THROW(scaler.transform(X), std::runtime_error);
    ASSERT_THROW(scaler.inverse_transform(X), std::runtime_error);
    ASSERT_THROW(scaler.mean(), std::runtime_error);
    ASSERT_THROW(scaler.var(), std::runtime_error);
    ASSERT_THROW(scaler.scale(), std::runtime_error);
}

void test_float_scalar() {
    Eigen::MatrixXf X(3, 2);
    X << 1, 2,
         3, 4,
         5, 6;

    Skigen::StandardScaler<float> scaler;
    scaler.fit(X);
    Eigen::MatrixXf Z = scaler.transform(X);
    Eigen::MatrixXf X_back = scaler.inverse_transform(Z);

    ASSERT_MATRIX_NEAR(X, X_back, 1e-5f);
}

void test_concept_satisfaction() {
    // Compile-time check: StandardScaler satisfies TransformerLike
    static_assert(Skigen::TransformerLike<Skigen::StandardScaler<double>>);
    static_assert(Skigen::TransformerLike<Skigen::StandardScaler<float>>);
    ASSERT_TRUE(true);  // If we got here, concepts are satisfied
}

void test_inverse_transform_inplace() {
    Eigen::MatrixXd X(3, 2);
    X << 1, 2,
         3, 4,
         5, 6;

    Skigen::StandardScaler scaler;
    scaler.fit(X);

    Eigen::MatrixXd Z = scaler.transform(X);
    Eigen::MatrixXd Z_copy = Z;

    scaler.inverse_transform_inplace(Z_copy);
    ASSERT_MATRIX_NEAR(X, Z_copy, 1e-12);
}

// ---------------------------------------------------------------------------
// partial_fit (Chan / Welford online update)
// ---------------------------------------------------------------------------

void test_partial_fit_first_call_equals_fit() {
    Eigen::MatrixXd X(6, 3);
    X << 1, 2, 3,
         4, 5, 6,
         7, 8, 9,
        10,11,12,
        13,14,15,
        16,17,18;

    Skigen::StandardScaler<double> a, b;
    a.fit(X);
    b.partial_fit(X);

    ASSERT_MATRIX_NEAR(a.mean(), b.mean(), 1e-12);
    ASSERT_MATRIX_NEAR(a.var(),  b.var(),  1e-12);
    ASSERT_MATRIX_NEAR(a.scale(), b.scale(), 1e-12);
    ASSERT_TRUE(a.n_samples_seen() == b.n_samples_seen());
}

void test_partial_fit_batched_equals_monolithic() {
    Eigen::MatrixXd X(9, 2);
    X << 1, 10,
         2, 20,
         3, 30,
         4, 40,
         5, 50,
         6, 60,
         7, 70,
         8, 80,
         9, 90;

    // Single fit on the full data
    Skigen::StandardScaler<double> mono;
    mono.fit(X);

    // Streaming partial_fit on three batches
    Skigen::StandardScaler<double> stream;
    stream.partial_fit(X.topRows(3));
    stream.partial_fit(X.middleRows(3, 4));
    stream.partial_fit(X.bottomRows(2));

    ASSERT_MATRIX_NEAR(mono.mean(),  stream.mean(),  1e-12);
    ASSERT_MATRIX_NEAR(mono.var(),   stream.var(),   1e-12);
    ASSERT_MATRIX_NEAR(mono.scale(), stream.scale(), 1e-12);
    ASSERT_TRUE(mono.n_samples_seen() == stream.n_samples_seen());

    // Verify transform also matches.
    ASSERT_MATRIX_NEAR(mono.transform(X), stream.transform(X), 1e-12);
}

void test_partial_fit_feature_mismatch_throws() {
    Eigen::MatrixXd X1(4, 3);
    X1 << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
    Eigen::MatrixXd X2(2, 4);
    X2.setOnes();

    Skigen::StandardScaler<double> sc;
    sc.partial_fit(X1);
    bool threw = false;
    try { sc.partial_fit(X2); }
    catch (const std::invalid_argument&) { threw = true; }
    ASSERT_TRUE(threw);
}

void test_partial_fit_without_std() {
    Eigen::MatrixXd X(6, 2);
    X << 1, 10, 2, 20, 3, 30, 4, 40, 5, 50, 6, 60;

    Skigen::StandardScaler<double> sc(/*with_mean=*/true, /*with_std=*/false);
    sc.partial_fit(X.topRows(3));
    sc.partial_fit(X.bottomRows(3));

    Skigen::StandardScaler<double> ref(true, false);
    ref.fit(X);
    ASSERT_MATRIX_NEAR(sc.mean(), ref.mean(), 1e-12);
    // var/scale stay at the with_std=false defaults (ones).
    ASSERT_MATRIX_NEAR(sc.var(),   Eigen::RowVectorXd::Ones(2), 1e-12);
    ASSERT_MATRIX_NEAR(sc.scale(), Eigen::RowVectorXd::Ones(2), 1e-12);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    std::cout << "=== StandardScaler Tests ===\n";

    run_test("fit_known_data",             test_fit_known_data);
    run_test("transform_correct_zscores",  test_transform_correct_zscores);
    run_test("round_trip",                 test_round_trip);
    run_test("fit_transform_equivalence",  test_fit_transform_equivalence);
    run_test("near_zero_variance",         test_near_zero_variance);
    run_test("without_mean",               test_without_mean);
    run_test("without_std",                test_without_std);
    run_test("transform_inplace",          test_transform_inplace);
    run_test("single_feature",             test_single_feature);
    run_test("single_sample",              test_single_sample);
    run_test("feature_mismatch_throws",    test_feature_mismatch_throws);
    run_test("not_fitted_throws",          test_not_fitted_throws);
    run_test("float_scalar",               test_float_scalar);
    run_test("concept_satisfaction",        test_concept_satisfaction);
    run_test("inverse_transform_inplace",  test_inverse_transform_inplace);
    run_test("partial_fit_first_call_equals_fit",       test_partial_fit_first_call_equals_fit);
    run_test("partial_fit_batched_equals_monolithic",   test_partial_fit_batched_equals_monolithic);
    run_test("partial_fit_feature_mismatch_throws",     test_partial_fit_feature_mismatch_throws);
    run_test("partial_fit_without_std",                 test_partial_fit_without_std);

    std::cout << "\n" << g_passed << " passed, " << g_failed << " failed.\n";
    return g_failed > 0 ? 1 : 0;
}
