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

static Eigen::MatrixXd make_input() {
    Eigen::MatrixXd input(8, 3);
    input << -2.0,  1.0, -1.30,
             -1.0,  0.2, -0.56,
              0.0, -0.4,  0.12,
              1.0,  0.7,  0.29,
              2.0,  1.5,  0.55,
              3.0, -1.2,  1.86,
              4.0,  0.4,  1.88,
              5.0, -0.7,  2.71;
    return input;
}

static Eigen::MatrixXd make_target(const Eigen::MatrixXd& input) {
    Eigen::MatrixXd target(input.rows(), 2);
    target.col(0) = (1.4 * input.col(0).array() - 0.6 * input.col(1).array() +
                     0.2 * input.col(2).array()).matrix();
    target.col(1) = (-0.3 * input.col(0).array() + 0.9 * input.col(1).array() +
                     0.8 * input.col(2).array()).matrix();
    return target;
}

static double correlation(const Eigen::VectorXd& left, const Eigen::VectorXd& right) {
    const Eigen::ArrayXd left_centered = left.array() - left.mean();
    const Eigen::ArrayXd right_centered = right.array() - right.mean();
    const double denom = std::sqrt(left_centered.square().sum() * right_centered.square().sum());
    return denom > 0.0 ? (left_centered * right_centered).sum() / denom : 0.0;
}

void test_pls_regression_predicts_linear_targets() {
    const Eigen::MatrixXd input = make_input();
    const Eigen::MatrixXd target = make_target(input);

    Skigen::PLSRegression<double> pls(/*n_components=*/2);
    pls.fit(input, target);
    const Eigen::MatrixXd predictions = pls.predict(input);

    ASSERT_TRUE(predictions.rows() == target.rows());
    ASSERT_TRUE(predictions.cols() == target.cols());
    ASSERT_TRUE(pls.x_weights().rows() == input.cols());
    ASSERT_TRUE(pls.y_weights().rows() == target.cols());
    ASSERT_TRUE(pls.x_scores().cols() == 2);
    ASSERT_TRUE(pls.coef().rows() == input.cols());
    ASSERT_TRUE(pls.coef().cols() == target.cols());
    ASSERT_TRUE(pls.n_iter().size() == 2);
    ASSERT_TRUE((pls.n_iter().array() > 0).all());
    ASSERT_TRUE(pls.score(input, target) > 0.99);
    ASSERT_TRUE((predictions - target).norm() < 0.5);
}

void test_pls_single_target_and_parameter_reflection() {
    const Eigen::MatrixXd input = make_input();
    Eigen::MatrixXd target(input.rows(), 1);
    target.col(0) = make_target(input).col(0);

    Skigen::PLSRegression<double> pls(/*n_components=*/1, /*scale=*/false, 100, 1e-8);
    pls.set_param("max_iter", Skigen::ParameterValue(200));
    pls.fit(input, target);

    const auto params = pls.get_params();
    ASSERT_TRUE(std::get<int>(params.at("max_iter")) == 200);
    ASSERT_TRUE(std::get<bool>(params.at("scale")) == false);
    ASSERT_TRUE(pls.predict(input).cols() == 1);
    ASSERT_TRUE(pls.transform(input).cols() == 1);
    ASSERT_TRUE(pls.score(input, target) > 0.95);
}

void test_cca_finds_correlated_scores() {
    const Eigen::MatrixXd input = make_input();
    const Eigen::MatrixXd target = make_target(input);

    Skigen::CCA<double> cca(/*n_components=*/2);
    cca.fit(input, target);
    auto [input_scores, target_scores] = cca.transform(input, target);

    ASSERT_TRUE(input_scores.rows() == input.rows());
    ASSERT_TRUE(input_scores.cols() == 2);
    ASSERT_TRUE(target_scores.rows() == target.rows());
    ASSERT_TRUE(target_scores.cols() == 2);
    ASSERT_TRUE(cca.x_rotations().cols() == 2);
    ASSERT_TRUE(cca.y_rotations().cols() == 2);
    ASSERT_TRUE(std::abs(correlation(input_scores.col(0), target_scores.col(0))) > 0.95);
    ASSERT_TRUE(std::abs(correlation(input_scores.col(1), target_scores.col(1))) > 0.85);
    ASSERT_TRUE(cca.score(input, target) > 0.8);
}

void test_invalid_inputs_throw() {
    const Eigen::MatrixXd input = make_input();
    const Eigen::MatrixXd target = make_target(input);

    Skigen::PLSRegression<double> bad_components(/*n_components=*/0);
    ASSERT_THROW(bad_components.fit(input, target), std::invalid_argument);

    Skigen::CCA<double> too_many_components(/*n_components=*/3);
    ASSERT_THROW(too_many_components.fit(input, target), std::invalid_argument);

    Skigen::PLSRegression<double> pls;
    ASSERT_THROW(pls.predict(input), std::runtime_error);

    Eigen::MatrixXd wrong_target(input.rows() - 1, target.cols());
    wrong_target.setZero();
    ASSERT_THROW(pls.fit(input, wrong_target), std::invalid_argument);
}

int main() {
    std::cout << "Running CrossDecomposition tests...\n";
    run_test("PLSRegression predicts linear targets", test_pls_regression_predicts_linear_targets);
    run_test("PLSRegression single target and params", test_pls_single_target_and_parameter_reflection);
    run_test("CCA finds correlated scores", test_cca_finds_correlated_scores);
    run_test("invalid inputs throw", test_invalid_inputs_throw);

    std::cout << "\nPassed: " << g_passed << ", Failed: " << g_failed << "\n";
    return g_failed == 0 ? 0 : 1;
}
