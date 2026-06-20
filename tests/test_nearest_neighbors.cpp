// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

#include <Skigen/Dense>

#include <Eigen/Core>
#include <functional>
#include <iostream>
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

void test_kneighbors_indices_and_distances() {
    Eigen::MatrixXd X(4, 2);
    X << 0.0, 0.0,
         1.0, 0.0,
         0.0, 2.0,
         3.0, 0.0;
    Eigen::MatrixXd Q(1, 2);
    Q << 0.2, 0.0;

    Skigen::NearestNeighbors<double> nn(2);
    nn.fit(X);
    Eigen::MatrixXi idx = nn.kneighbors(Q);
    Eigen::MatrixXd dist = nn.kneighbors_distances(Q);

    ASSERT_TRUE(idx.rows() == 1);
    ASSERT_TRUE(idx.cols() == 2);
    ASSERT_TRUE(idx(0, 0) == 0);
    ASSERT_TRUE(idx(0, 1) == 1);
    ASSERT_TRUE(dist(0, 0) <= dist(0, 1));
}

void test_radius_neighbors() {
    Eigen::MatrixXd X(4, 1);
    X << 0.0, 0.5, 2.0, 3.0;
    Eigen::MatrixXd Q(1, 1);
    Q << 0.25;

    Skigen::NearestNeighbors<double> nn(2, 0.5);
    nn.fit(X);
    auto neighbors = nn.radius_neighbors(Q);
    ASSERT_TRUE(neighbors.size() == 1);
    ASSERT_TRUE(neighbors[0].size() == 2);
    ASSERT_TRUE(neighbors[0][0] == 0);
    ASSERT_TRUE(neighbors[0][1] == 1);
}

void test_errors() {
    Eigen::MatrixXd X(2, 2);
    X << 0.0, 0.0,
         1.0, 1.0;
    Eigen::MatrixXd Q(1, 2);
    Q << 0.0, 0.0;

    Skigen::NearestNeighbors<double> nn(3);
    ASSERT_THROW(nn.fit(X), std::invalid_argument);
    Skigen::NearestNeighbors<double> ok(1);
    ASSERT_THROW(ok.kneighbors(Q), std::runtime_error);
    ok.fit(X);
    Eigen::MatrixXd wrong(1, 3);
    wrong << 0.0, 0.0, 0.0;
    ASSERT_THROW(ok.kneighbors(wrong), std::invalid_argument);
}

int main() {
    std::cout << "Running NearestNeighbors tests...\n";
    run_test("kneighbors indices and distances", test_kneighbors_indices_and_distances);
    run_test("radius neighbors", test_radius_neighbors);
    run_test("errors", test_errors);

    std::cout << "\nPassed: " << g_passed << ", Failed: " << g_failed << "\n";
    return g_failed == 0 ? 0 : 1;
}
