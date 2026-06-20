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

Eigen::MatrixXd make_classification_X() {
    Eigen::MatrixXd X(8, 2);
    X << -2.0, -1.0,
         -1.6, -1.5,
         -1.2, -0.8,
         -1.8, -1.1,
          1.0,  1.7,
          1.4,  1.1,
          1.8,  1.6,
          1.2,  1.3;
    return X;
}

Eigen::VectorXi make_classification_y() {
    Eigen::VectorXi y(8);
    y << 0, 0, 0, 0, 1, 1, 1, 1;
    return y;
}

void test_lda_fit_predict_proba() {
    const auto X = make_classification_X();
    const auto y = make_classification_y();

    Skigen::LinearDiscriminantAnalysis<double> lda;
    lda.fit(X, y);
    const Eigen::VectorXi pred = lda.predict(X);
    for (Eigen::Index i = 0; i < y.size(); ++i) {
        ASSERT_TRUE(pred(i) == y(i));
    }

    ASSERT_TRUE(lda.classes().size() == 2);
    ASSERT_NEAR(lda.priors()(0), 0.5, 1e-12);
    ASSERT_NEAR(lda.priors()(1), 0.5, 1e-12);
    ASSERT_TRUE(lda.means().rows() == 2);
    ASSERT_TRUE(lda.coef().cols() == X.cols());

    const Eigen::MatrixXd P = lda.predict_proba(X);
    ASSERT_TRUE(P.rows() == X.rows());
    ASSERT_TRUE(P.cols() == 2);
    for (Eigen::Index row = 0; row < P.rows(); ++row) {
        ASSERT_NEAR(P.row(row).sum(), 1.0, 1e-12);
    }
}

void test_qda_fit_predict_proba() {
    const auto X = make_classification_X();
    const auto y = make_classification_y();

    Skigen::QuadraticDiscriminantAnalysis<double> qda(1e-6);
    qda.fit(X, y);
    const Eigen::VectorXi pred = qda.predict(X);
    for (Eigen::Index i = 0; i < y.size(); ++i) {
        ASSERT_TRUE(pred(i) == y(i));
    }

    ASSERT_TRUE(qda.covariance().size() == 2);
    const Eigen::MatrixXd P = qda.predict_proba(X);
    for (Eigen::Index row = 0; row < P.rows(); ++row) {
        ASSERT_NEAR(P.row(row).sum(), 1.0, 1e-12);
    }
}

void test_log_proba_consistency() {
    const auto X = make_classification_X();
    const auto y = make_classification_y();

    Skigen::LinearDiscriminantAnalysis<double> lda;
    lda.fit(X, y);
    const Eigen::MatrixXd P = lda.predict_proba(X);
    const Eigen::MatrixXd LP = lda.predict_log_proba(X);
    for (Eigen::Index row = 0; row < P.rows(); ++row) {
        for (Eigen::Index col = 0; col < P.cols(); ++col) {
            ASSERT_NEAR(std::exp(LP(row, col)), P(row, col), 1e-12);
        }
    }
}

void test_invalid_inputs() {
    Eigen::MatrixXd X(2, 2);
    X << 1.0, 2.0,
         3.0, 4.0;
    Eigen::VectorXi one_class(2);
    one_class << 0, 0;

    Skigen::LinearDiscriminantAnalysis<double> lda;
    ASSERT_THROW(lda.fit(X, one_class), std::invalid_argument);

    Eigen::VectorXi y(2);
    y << 0, 1;
    Skigen::QuadraticDiscriminantAnalysis<double> qda;
    ASSERT_THROW(qda.fit(X, y), std::invalid_argument);

    Skigen::QuadraticDiscriminantAnalysis<double> bad(1.5);
    ASSERT_THROW(bad.fit(X, y), std::invalid_argument);
}

int main() {
    std::cout << "Running DiscriminantAnalysis tests...\n";
    run_test("LDA fit/predict/proba", test_lda_fit_predict_proba);
    run_test("QDA fit/predict/proba", test_qda_fit_predict_proba);
    run_test("log-proba consistency", test_log_proba_consistency);
    run_test("invalid inputs", test_invalid_inputs);

    std::cout << "\nPassed: " << g_passed << ", Failed: " << g_failed << "\n";
    return g_failed == 0 ? 0 : 1;
}
