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
void assert_near(Scalar actual, Scalar expected, Scalar tol, const char* file, int line) {
    if (std::abs(actual - expected) > tol) {
        std::ostringstream oss;
        oss << file << ":" << line << ": ASSERT_NEAR failed: " << actual
            << " vs " << expected << " (tol=" << tol << ")";
        throw TestFailure(oss.str());
    }
}
#define ASSERT_NEAR(actual, expected, tol) assert_near(actual, expected, tol, __FILE__, __LINE__)

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

static Eigen::MatrixXd make_outlier_fixture() {
    Eigen::MatrixXd input(12, 2);
    input << -0.6, -0.4,
             -0.3,  0.2,
              0.0, -0.1,
              0.4,  0.3,
              0.6, -0.2,
             -0.5,  0.5,
              0.2,  0.6,
              0.5,  0.1,
             -0.2, -0.6,
              0.1,  0.4,
              5.0,  5.5,
             -5.0,  4.5;
    return input;
}

void test_elliptic_envelope_scores_outliers_lower() {
    const Eigen::MatrixXd input = make_outlier_fixture();
    Skigen::EllipticEnvelope<double> envelope(/*contamination=*/1.0 / 6.0);
    envelope.fit(input);

    const Eigen::VectorXd scores = envelope.score_samples(input);
    const Eigen::VectorXi labels = envelope.predict(input);
    ASSERT_TRUE(envelope.is_fitted());
    ASSERT_TRUE(envelope.location().cols() == input.cols());
    ASSERT_TRUE(envelope.covariance().rows() == input.cols());
    ASSERT_TRUE(envelope.precision().cols() == input.cols());
    ASSERT_TRUE(envelope.dist().size() == input.rows());
    ASSERT_TRUE(scores(10) < scores(0));
    ASSERT_TRUE(scores(11) < scores(1));
    ASSERT_TRUE(labels(10) == -1);
    ASSERT_TRUE(labels(11) == -1);
    ASSERT_TRUE((labels.array() == -1).count() == 2);
}

void test_elliptic_envelope_decision_threshold() {
    const Eigen::MatrixXd input = make_outlier_fixture();
    Skigen::EllipticEnvelope<double> envelope(/*contamination=*/0.25);
    envelope.fit(input);

    const Eigen::VectorXd scores = envelope.score_samples(input);
    const Eigen::VectorXd decisions = envelope.decision_function(input);
    ASSERT_TRUE(scores.allFinite());
    ASSERT_TRUE(decisions.allFinite());
    ASSERT_NEAR(decisions(0), scores(0) - envelope.offset(), 1e-12);
    ASSERT_TRUE((envelope.predict(input).array() == -1).count() == 3);
}

void test_isolation_forest_deterministic_and_separates_outliers() {
    const Eigen::MatrixXd input = make_outlier_fixture();
    Skigen::IsolationForest<double> first(
        /*n_estimators=*/96,
        /*max_samples=*/8,
        /*contamination=*/1.0 / 6.0,
        /*max_depth=*/-1,
        /*random_state=*/11);
    Skigen::IsolationForest<double> second(
        /*n_estimators=*/96,
        /*max_samples=*/8,
        /*contamination=*/1.0 / 6.0,
        /*max_depth=*/-1,
        /*random_state=*/11);
    first.fit(input);
    second.fit(input);

    const Eigen::VectorXd first_scores = first.score_samples(input);
    const Eigen::VectorXd second_scores = second.score_samples(input);
    const Eigen::VectorXi labels = first.predict(input);
    ASSERT_TRUE((first_scores - second_scores).norm() < 1e-12);
    ASSERT_TRUE(first.max_samples_effective() == 8);
    ASSERT_TRUE(first_scores(10) < first_scores(0));
    ASSERT_TRUE(first_scores(11) < first_scores(1));
    ASSERT_TRUE(labels(10) == -1);
    ASSERT_TRUE(labels(11) == -1);
    ASSERT_TRUE((labels.array() == -1).count() == 2);
}

void test_invalid_inputs_throw() {
    const Eigen::MatrixXd input = make_outlier_fixture();

    Skigen::EllipticEnvelope<double> bad_contamination(/*contamination=*/0.8);
    ASSERT_THROW(bad_contamination.fit(input), std::invalid_argument);

    Skigen::EllipticEnvelope<double> envelope;
    ASSERT_THROW(envelope.predict(input), std::runtime_error);

    Skigen::IsolationForest<double> bad_estimators(/*n_estimators=*/0);
    ASSERT_THROW(bad_estimators.fit(input), std::invalid_argument);

    Skigen::IsolationForest<double> bad_samples(
        /*n_estimators=*/10,
        /*max_samples=*/20);
    ASSERT_THROW(bad_samples.fit(input), std::invalid_argument);
}

int main() {
    std::cout << "Running Anomaly tests...\n";
    run_test("EllipticEnvelope scores outliers lower", test_elliptic_envelope_scores_outliers_lower);
    run_test("EllipticEnvelope decision threshold", test_elliptic_envelope_decision_threshold);
    run_test("IsolationForest deterministic separation", test_isolation_forest_deterministic_and_separates_outliers);
    run_test("invalid inputs throw", test_invalid_inputs_throw);

    std::cout << "\nPassed: " << g_passed << ", Failed: " << g_failed << "\n";
    return g_failed == 0 ? 0 : 1;
}
