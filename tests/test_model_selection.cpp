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
// TrainTestSplit Tests
// ===================================================================

void test_tts_basic() {
    Eigen::MatrixXd X(10, 2);
    Eigen::VectorXd y(10);
    for (int i = 0; i < 10; ++i) {
        X(i, 0) = static_cast<double>(i);
        X(i, 1) = static_cast<double>(i) * 2;
        y(i) = static_cast<double>(i);
    }

    auto result = Skigen::train_test_split(X, y, 0.3);

    // 30% of 10 = 3 test, 7 train
    ASSERT_TRUE(result.X_train.rows() == 7);
    ASSERT_TRUE(result.X_test.rows() == 3);
    ASSERT_TRUE(result.y_train.size() == 7);
    ASSERT_TRUE(result.y_test.size() == 3);
    ASSERT_TRUE(result.X_train.cols() == 2);
    ASSERT_TRUE(result.X_test.cols() == 2);
}

void test_tts_preserves_data() {
    Eigen::MatrixXd X(4, 1);
    X << 10, 20, 30, 40;
    Eigen::VectorXd y(4);
    y << 1, 2, 3, 4;

    auto result = Skigen::train_test_split(X, y, 0.25);

    // Total samples preserved
    ASSERT_TRUE(result.X_train.rows() + result.X_test.rows() == 4);
}

void test_tts_deterministic() {
    Eigen::MatrixXd X(10, 2);
    Eigen::VectorXd y(10);
    for (int i = 0; i < 10; ++i) {
        X(i, 0) = static_cast<double>(i);
        X(i, 1) = static_cast<double>(i);
        y(i) = static_cast<double>(i);
    }

    auto r1 = Skigen::train_test_split(X, y, 0.3, 42);
    auto r2 = Skigen::train_test_split(X, y, 0.3, 42);

    // Same seed → same split
    for (Eigen::Index i = 0; i < r1.y_train.size(); ++i) {
        ASSERT_NEAR(r1.y_train(i), r2.y_train(i), 1e-10);
    }
}

void test_tts_invalid_size() {
    Eigen::MatrixXd X(4, 1);
    X << 1, 2, 3, 4;
    Eigen::VectorXd y(4);
    y << 1, 2, 3, 4;

    ASSERT_THROW(Skigen::train_test_split(X, y, 0.0), std::invalid_argument);
    ASSERT_THROW(Skigen::train_test_split(X, y, 1.0), std::invalid_argument);
}

void test_tts_inconsistent_lengths() {
    Eigen::MatrixXd X(4, 1);
    X << 1, 2, 3, 4;
    Eigen::VectorXd y(3);
    y << 1, 2, 3;

    ASSERT_THROW(Skigen::train_test_split(X, y, 0.25), std::invalid_argument);
}

// ===================================================================

int main() {
    std::cout << "=== TrainTestSplit Tests ===\n";
    run_test("tts_basic", test_tts_basic);
    run_test("tts_preserves_data", test_tts_preserves_data);
    run_test("tts_deterministic", test_tts_deterministic);
    run_test("tts_invalid_size", test_tts_invalid_size);
    run_test("tts_inconsistent_lengths", test_tts_inconsistent_lengths);

    std::cout << "\n" << g_passed << " passed, " << g_failed << " failed.\n";
    return g_failed > 0 ? 1 : 0;
}
