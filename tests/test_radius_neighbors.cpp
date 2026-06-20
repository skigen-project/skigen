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

void test_radius_classifier_majority_vote() {
    Eigen::MatrixXd X(5, 1);
    X << 0.0, 0.2, 0.4, 2.0, 2.2;
    Eigen::VectorXi y(5);
    y << 0, 0, 1, 1, 1;
    Eigen::MatrixXd Q(2, 1);
    Q << 0.1, 2.1;

    Skigen::RadiusNeighborsClassifier<double> clf(0.35);
    clf.fit(X, y);
    Eigen::VectorXi pred = clf.predict(Q);
    ASSERT_TRUE(pred(0) == 0);
    ASSERT_TRUE(pred(1) == 1);
}

void test_radius_regressor_mean() {
    Eigen::MatrixXd X(4, 1);
    X << 0.0, 0.2, 1.0, 1.2;
    Eigen::VectorXd y(4);
    y << 1.0, 3.0, 10.0, 14.0;
    Eigen::MatrixXd Q(2, 1);
    Q << 0.1, 1.1;

    Skigen::RadiusNeighborsRegressor<double> reg(0.25);
    reg.fit(X, y);
    Eigen::VectorXd pred = reg.predict(Q);
    ASSERT_TRUE(std::abs(pred(0) - 2.0) < 1e-12);
    ASSERT_TRUE(std::abs(pred(1) - 12.0) < 1e-12);
}

void test_radius_neighbors_lists() {
    Eigen::MatrixXd X(4, 2);
    X << 0.0, 0.0,
         0.5, 0.0,
         2.0, 0.0,
         3.0, 0.0;
    Eigen::VectorXi y(4);
    y << 0, 0, 1, 1;
    Eigen::MatrixXd Q(1, 2);
    Q << 0.25, 0.0;

    Skigen::RadiusNeighborsClassifier<double> clf(0.5);
    clf.fit(X, y);
    auto neighbors = clf.radius_neighbors(Q);
    ASSERT_TRUE(neighbors.size() == 1);
    ASSERT_TRUE(neighbors[0].size() == 2);
    ASSERT_TRUE(neighbors[0][0] == 0);
    ASSERT_TRUE(neighbors[0][1] == 1);
}

void test_radius_errors() {
    Eigen::MatrixXd X(2, 1);
    X << 0.0, 1.0;
    Eigen::VectorXi yc(2);
    yc << 0, 1;
    Eigen::VectorXd yr(2);
    yr << 0.0, 1.0;
    Eigen::MatrixXd far(1, 1);
    far << 10.0;

    Skigen::RadiusNeighborsClassifier<double> clf(0.1);
    clf.fit(X, yc);
    ASSERT_THROW(clf.predict(far), std::runtime_error);

    Skigen::RadiusNeighborsRegressor<double> reg(0.1);
    reg.fit(X, yr);
    ASSERT_THROW(reg.predict(far), std::runtime_error);

    Skigen::RadiusNeighborsClassifier<double> bad(-1.0);
    ASSERT_THROW(bad.fit(X, yc), std::invalid_argument);
}

int main() {
    std::cout << "Running RadiusNeighbors tests...\n";
    run_test("classifier majority vote", test_radius_classifier_majority_vote);
    run_test("regressor mean", test_radius_regressor_mean);
    run_test("radius neighbor lists", test_radius_neighbors_lists);
    run_test("radius errors", test_radius_errors);

    std::cout << "\nPassed: " << g_passed << ", Failed: " << g_failed << "\n";
    return g_failed == 0 ? 0 : 1;
}
