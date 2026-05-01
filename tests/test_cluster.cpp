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
// KMeans Tests
// ===================================================================

void test_kmeans_basic() {
    // Two well-separated clusters
    Eigen::MatrixXd X(6, 2);
    X << 1, 1,
         1.5, 1.5,
         1, 1.5,
         10, 10,
         10.5, 10.5,
         10, 10.5;

    Skigen::KMeans<double> km(2);
    km.fit(X);

    ASSERT_TRUE(km.is_fitted());
    ASSERT_TRUE(km.cluster_centers().rows() == 2);
    ASSERT_TRUE(km.labels().size() == 6);

    // First 3 and last 3 should be in different clusters
    ASSERT_TRUE(km.labels()(0) == km.labels()(1));
    ASSERT_TRUE(km.labels()(0) == km.labels()(2));
    ASSERT_TRUE(km.labels()(3) == km.labels()(4));
    ASSERT_TRUE(km.labels()(3) == km.labels()(5));
    ASSERT_TRUE(km.labels()(0) != km.labels()(3));
}

void test_kmeans_predict() {
    Eigen::MatrixXd X(4, 2);
    X << 0, 0,
         0, 1,
         10, 10,
         10, 11;

    Skigen::KMeans<double> km(2);
    km.fit(X);

    Eigen::MatrixXd X_new(2, 2);
    X_new << 0.5, 0.5,
             9.5, 10.0;

    auto labels = km.predict(X_new);
    ASSERT_TRUE(labels(0) != labels(1));
}

void test_kmeans_transform() {
    Eigen::MatrixXd X(4, 2);
    X << 0, 0, 0, 1, 10, 10, 10, 11;

    Skigen::KMeans<double> km(2);
    km.fit(X);

    auto dists = km.transform(X);
    ASSERT_TRUE(dists.rows() == 4);
    ASSERT_TRUE(dists.cols() == 2);
}

void test_kmeans_inertia() {
    // Perfect clusters
    Eigen::MatrixXd X(2, 2);
    X << 0, 0,
         10, 10;

    Skigen::KMeans<double> km(2);
    km.fit(X);

    // Each point is its own center, inertia should be very small
    ASSERT_NEAR(km.inertia(), 0.0, 1e-10);
}

void test_kmeans_not_fitted() {
    Skigen::KMeans<double> km;
    Eigen::MatrixXd X(3, 2);
    X << 1, 2, 3, 4, 5, 6;
    ASSERT_THROW(km.predict(X), std::runtime_error);
}

// ===================================================================

int main() {
    std::cout << "=== KMeans Tests ===\n";
    run_test("kmeans_basic", test_kmeans_basic);
    run_test("kmeans_predict", test_kmeans_predict);
    run_test("kmeans_transform", test_kmeans_transform);
    run_test("kmeans_inertia", test_kmeans_inertia);
    run_test("kmeans_not_fitted", test_kmeans_not_fitted);

    std::cout << "\n" << g_passed << " passed, " << g_failed << " failed.\n";
    return g_failed > 0 ? 1 : 0;
}
