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
#include <vector>

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

template <typename VecA, typename VecB, typename Scalar>
void assert_vector_near(const VecA& a, const VecB& b, Scalar tol,
                        const char* file, int line) {
    if (a.size() != b.size()) {
        throw TestFailure(std::string(file) + ":" + std::to_string(line) +
                          ": vector size mismatch");
    }
    for (Eigen::Index i = 0; i < a.size(); ++i) {
        assert_near(a(i), b(i), tol, file, line);
    }
}
#define ASSERT_VECTOR_NEAR(a, b, tol) assert_vector_near(a, b, tol, __FILE__, __LINE__)

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

void test_rbf_interpolates_training_data() {
    Eigen::MatrixXd X(5, 1);
    X << -2.0, -1.0, 0.0, 1.0, 2.0;
    Eigen::VectorXd y(5);
    y << 0.5, -0.2, 0.0, 0.8, 1.5;

    Skigen::GaussianProcessRegressor<double> gpr(
        Skigen::GaussianProcessRegressor<double>::Kernel::RBF,
        1e-8,
        0.9,
        1.0,
        1.0,
        1.5,
        1.0,
        1.0,
        1.0,
        true);
    gpr.fit(X, y);

    ASSERT_VECTOR_NEAR(gpr.predict(X), y, 1e-5);
    const Eigen::VectorXd std = gpr.predict_std(X);
    ASSERT_TRUE((std.array() < 5e-4).all());
    ASSERT_TRUE(std.allFinite());
    ASSERT_TRUE(std.size() == X.rows());
    ASSERT_TRUE(gpr.dual_coef().size() == X.rows());
    ASSERT_TRUE(gpr.L().rows() == X.rows());
    ASSERT_TRUE(std::isfinite(gpr.log_marginal_likelihood_value()));
    ASSERT_NEAR(gpr.y_train_mean(), y.mean(), 1e-12);
}

void test_predict_covariance_is_symmetric() {
    Eigen::MatrixXd X(4, 2);
    X << 0.0, 0.0,
         1.0, 0.0,
         0.0, 1.0,
         1.0, 1.0;
    Eigen::VectorXd y(4);
    y << 0.0, 1.0, 1.0, 2.0;
    Eigen::MatrixXd X_test(2, 2);
    X_test << 0.25, 0.25,
              1.5, 1.5;

    Skigen::GaussianProcessRegressor<double> gpr(
        Skigen::GaussianProcessRegressor<double>::Kernel::RationalQuadratic,
        1e-6,
        1.2,
        1.5,
        1.0,
        1.5,
        0.8);
    gpr.fit(X, y);

    const Eigen::MatrixXd covariance = gpr.predict_covariance(X_test);
    ASSERT_TRUE(covariance.rows() == X_test.rows());
    ASSERT_TRUE(covariance.cols() == X_test.rows());
    ASSERT_TRUE(covariance.allFinite());
    ASSERT_NEAR(covariance(0, 1), covariance(1, 0), 1e-12);
    ASSERT_TRUE(covariance(0, 0) >= 0.0);
    ASSERT_TRUE(covariance(1, 1) >= 0.0);
}

void test_supported_kernels_are_finite() {
    using GPR = Skigen::GaussianProcessRegressor<double>;
    Eigen::MatrixXd X(4, 2);
    X << 1.0, 0.0,
         0.0, 1.0,
         1.0, 1.0,
         2.0, 1.0;
    Eigen::VectorXd y(4);
    y << 1.0, 0.5, 1.5, 2.0;

    const std::vector<GPR::Kernel> kernels = {
        GPR::Kernel::RBF,
        GPR::Kernel::Matern,
        GPR::Kernel::RationalQuadratic,
        GPR::Kernel::ExpSineSquared,
        GPR::Kernel::DotProduct,
        GPR::Kernel::White,
        GPR::Kernel::Constant};

    for (const GPR::Kernel kernel : kernels) {
        GPR gpr(kernel, 0.1, 1.0, 1.0, 0.5, 1.5, 1.0, 2.0, 0.3);
        gpr.fit(X, y);
        ASSERT_TRUE(gpr.predict(X).allFinite());
        ASSERT_TRUE(gpr.predict_std(X).allFinite());
    }
}

void test_invalid_and_unfitted_errors() {
    Eigen::MatrixXd X(2, 2);
    X << 1.0, 2.0,
         3.0, 4.0;
    Eigen::VectorXd y(2);
    y << 1.0, 2.0;

    Skigen::GaussianProcessRegressor<double> gpr;
    ASSERT_THROW(gpr.predict(X), std::runtime_error);

    Skigen::GaussianProcessRegressor<double> bad_alpha(
        Skigen::GaussianProcessRegressor<double>::Kernel::RBF,
        -1.0);
    ASSERT_THROW(bad_alpha.fit(X, y), std::invalid_argument);

    Skigen::GaussianProcessRegressor<double> bad_matern(
        Skigen::GaussianProcessRegressor<double>::Kernel::Matern,
        1e-6,
        1.0,
        1.0,
        1.0,
        0.7);
    ASSERT_THROW(bad_matern.fit(X, y), std::invalid_argument);

    gpr.fit(X, y);
    Eigen::MatrixXd wrong(1, 3);
    wrong << 1.0, 2.0, 3.0;
    ASSERT_THROW(gpr.predict(wrong), std::invalid_argument);
}

void test_classifier_binary_probabilities() {
    Eigen::MatrixXd X(8, 2);
    X << -2.0, -1.0,
         -1.5, -0.6,
         -1.0, -1.2,
         -0.7, -0.3,
          0.7,  0.4,
          1.0,  1.1,
          1.4,  0.5,
          2.0,  1.2;
    Eigen::VectorXi y(8);
    y << 0, 0, 0, 0, 1, 1, 1, 1;

    Skigen::GaussianProcessClassifier<double> gpc(
        Skigen::GaussianProcessClassifier<double>::Kernel::RBF,
        1e-6,
        1.0,
        1.0,
        1.0,
        1.5,
        1.0,
        1.0,
        1.0,
        50,
        1e-7);
    gpc.fit(X, y);

    const Eigen::VectorXi predictions = gpc.predict(X);
    const Eigen::MatrixXd probabilities = gpc.predict_proba(X);
    ASSERT_TRUE(gpc.classes().size() == 2);
    ASSERT_TRUE(gpc.classes()(0) == 0);
    ASSERT_TRUE(gpc.classes()(1) == 1);
    ASSERT_TRUE(probabilities.rows() == X.rows());
    ASSERT_TRUE(probabilities.cols() == 2);
    ASSERT_TRUE(probabilities.allFinite());
    ASSERT_TRUE(gpc.latent_mean().size() == X.rows());
    ASSERT_TRUE(gpc.dual_coef().size() == X.rows());
    ASSERT_TRUE(gpc.n_iter() > 0);
    for (Eigen::Index row = 0; row < probabilities.rows(); ++row) {
        ASSERT_NEAR(probabilities.row(row).sum(), 1.0, 1e-12);
        ASSERT_TRUE(probabilities(row, 0) >= 0.0 && probabilities(row, 0) <= 1.0);
        ASSERT_TRUE(probabilities(row, 1) >= 0.0 && probabilities(row, 1) <= 1.0);
    }
    ASSERT_TRUE((predictions.array() == y.array()).all());
    ASSERT_TRUE(gpc.score(X, y) == 1.0);
}

void test_classifier_invalid_inputs() {
    Eigen::MatrixXd X(4, 2);
    X << 0.0, 0.0,
         1.0, 0.0,
         0.0, 1.0,
         1.0, 1.0;
    Eigen::VectorXi three_classes(4);
    three_classes << 0, 1, 2, 1;

    Skigen::GaussianProcessClassifier<double> gpc;
    ASSERT_THROW(gpc.predict(X), std::runtime_error);
    ASSERT_THROW(gpc.fit(X, three_classes), std::invalid_argument);

    Skigen::GaussianProcessClassifier<double> bad_alpha(
        Skigen::GaussianProcessClassifier<double>::Kernel::RBF,
        -1.0);
    Eigen::VectorXi y(4);
    y << 0, 0, 1, 1;
    ASSERT_THROW(bad_alpha.fit(X, y), std::invalid_argument);
}

int main() {
    std::cout << "Running GaussianProcess tests...\n";
    run_test("RBF interpolates training data", test_rbf_interpolates_training_data);
    run_test("posterior covariance is symmetric", test_predict_covariance_is_symmetric);
    run_test("supported kernels are finite", test_supported_kernels_are_finite);
    run_test("invalid and unfitted errors", test_invalid_and_unfitted_errors);
    run_test("classifier binary probabilities", test_classifier_binary_probabilities);
    run_test("classifier invalid inputs", test_classifier_invalid_inputs);

    std::cout << "\nPassed: " << g_passed << ", Failed: " << g_failed << "\n";
    return g_failed == 0 ? 0 : 1;
}
