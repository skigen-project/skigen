// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#include <Skigen/Dense>

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <functional>

// ---------------------------------------------------------------------------
// Minimal test harness
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

template <typename VecA, typename VecB, typename Scalar>
void assert_vector_near(const VecA& A, const VecB& B, Scalar tol,
                        const char* file, int line) {
    if (A.size() != B.size()) {
        throw TestFailure(std::string(file) + ":" + std::to_string(line) +
                          ": Size mismatch");
    }
    for (Eigen::Index i = 0; i < A.size(); ++i)
        assert_near(A(i), B(i), tol, file, line);
}
#define ASSERT_VECTOR_NEAR(A, B, tol) assert_vector_near(A, B, tol, __FILE__, __LINE__)

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

// ===================================================================
// PCA Tests
// ===================================================================

void test_pca_basic() {
    // 2D data with strong first component
    Eigen::MatrixXd X(5, 2);
    X << 1, 2,
         3, 4,
         5, 6,
         7, 8,
         9, 10;

    Skigen::PCA pca(2);
    pca.fit(X);

    ASSERT_TRUE(pca.n_components() == 2);
    ASSERT_TRUE(pca.components().rows() == 2);
    ASSERT_TRUE(pca.components().cols() == 2);
    ASSERT_TRUE(pca.explained_variance().size() == 2);
}

void test_pca_variance_ratio_sums_to_one() {
    Eigen::MatrixXd X(5, 2);
    X << 1, 2,
         3, 4,
         5, 6,
         7, 8,
         9, 10;

    Skigen::PCA pca;
    pca.fit(X);

    double ratio_sum = pca.explained_variance_ratio().sum();
    ASSERT_NEAR(ratio_sum, 1.0, 1e-10);
}

void test_pca_dimensionality_reduction() {
    Eigen::MatrixXd X(5, 3);
    X << 1, 2, 3,
         4, 5, 6,
         7, 8, 9,
         10, 11, 12,
         13, 14, 15;

    Skigen::PCA pca(2);
    pca.fit(X);
    Eigen::MatrixXd Z = pca.transform(X);

    ASSERT_TRUE(Z.rows() == 5);
    ASSERT_TRUE(Z.cols() == 2);
}

void test_pca_inverse_transform() {
    Eigen::MatrixXd X(5, 2);
    X << 1, 2,
         3, 4,
         5, 6,
         7, 8,
         9, 10;

    Skigen::PCA pca(2);
    pca.fit(X);
    Eigen::MatrixXd Z = pca.transform(X);
    Eigen::MatrixXd X_hat = pca.inverse_transform(Z);

    // Perfect reconstruction with all components
    for (Eigen::Index i = 0; i < X.rows(); ++i) {
        for (Eigen::Index j = 0; j < X.cols(); ++j) {
            ASSERT_NEAR(X_hat(i, j), X(i, j), 1e-10);
        }
    }
}

void test_pca_fit_transform() {
    Eigen::MatrixXd X(5, 2);
    X << 1, 2,
         3, 4,
         5, 6,
         7, 8,
         9, 10;

    Skigen::PCA pca1(1);
    pca1.fit(X);
    Eigen::MatrixXd Z1 = pca1.transform(X);

    Skigen::PCA<double> pca2(1);
    Eigen::MatrixXd Z2 = pca2.fit_transform(X);

    for (Eigen::Index i = 0; i < Z1.rows(); ++i) {
        ASSERT_NEAR(std::abs(Z1(i, 0)), std::abs(Z2(i, 0)), 1e-10);
    }
}

void test_pca_not_fitted() {
    Skigen::PCA pca;
    Eigen::MatrixXd X(3, 2);
    X << 1, 2, 3, 4, 5, 6;
    ASSERT_THROW(pca.transform(X), std::runtime_error);
}

// ===================================================================

int main() {
    std::cout << "=== PCA Tests ===\n";
    run_test("pca_basic", test_pca_basic);
    run_test("pca_variance_ratio_sums_to_one", test_pca_variance_ratio_sums_to_one);
    run_test("pca_dimensionality_reduction", test_pca_dimensionality_reduction);
    run_test("pca_inverse_transform", test_pca_inverse_transform);
    run_test("pca_fit_transform", test_pca_fit_transform);
    run_test("pca_not_fitted", test_pca_not_fitted);

    std::cout << "\n" << g_passed << " passed, " << g_failed << " failed.\n";
    return g_failed > 0 ? 1 : 0;
}
