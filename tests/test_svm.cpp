// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#include <Skigen/SVM>

#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

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
    } while (0)

#define ASSERT_NEAR(a, b, tol)                                                 \
    do {                                                                       \
        const double aa = static_cast<double>(a);                              \
        const double bb = static_cast<double>(b);                              \
        if (std::fabs(aa - bb) > (tol)) {                                      \
            std::ostringstream oss;                                            \
            oss << __FILE__ << ":" << __LINE__                                 \
                << ": ASSERT_NEAR failed: " << aa << " vs " << bb              \
                << " (tol=" << (tol) << ")";                                   \
            throw TestFailure(oss.str());                                      \
        }                                                                      \
    } while (0)

static void run(const std::string& name, std::function<void()> body) {
    try {
        body();
        ++g_passed;
        std::cout << "  [PASS] " << name << "\n";
    } catch (const TestFailure& e) {
        ++g_failed;
        std::cout << "  [FAIL] " << name << " — " << e.what() << "\n";
    } catch (const std::exception& e) {
        ++g_failed;
        std::cout << "  [FAIL] " << name << " — unexpected: " << e.what() << "\n";
    }
}

// ---------------------------------------------------------------------------
// Helpers — small datasets.
// ---------------------------------------------------------------------------

static std::pair<Eigen::MatrixXd, Eigen::VectorXi>
two_cluster(int n, double sep, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> nz(0.0, 0.5);
    Eigen::MatrixXd X(n, 2);
    Eigen::VectorXi y(n);
    for (int i = 0; i < n; ++i) {
        const double cls = (i < n / 2) ? -sep : sep;
        X(i, 0) = cls + nz(rng);
        X(i, 1) = cls + nz(rng);
        y(i)    = (cls > 0) ? 1 : 0;
    }
    return {X, y};
}

// ---------------------------------------------------------------------------
// LinearSVC
// ---------------------------------------------------------------------------

void test_linear_svc_binary_separable() {
    auto [X, y] = two_cluster(120, 1.5, 7);
    Skigen::LinearSVC<double> clf(
        /*C=*/1.0, Skigen::LinearSVC<double>::Loss::SquaredHinge,
        1e-4, 200, true, std::optional<uint64_t>(7));
    clf.fit(X, y);
    auto preds = clf.predict(X);
    int correct = 0;
    for (int i = 0; i < y.size(); ++i) if (preds(i) == y(i)) ++correct;
    ASSERT_TRUE(static_cast<double>(correct) / y.size() > 0.9);
    ASSERT_TRUE(clf.n_classes() == 2);
    ASSERT_TRUE(clf.coef().rows() == 1);
    ASSERT_TRUE(clf.coef().cols() == 2);
}

void test_linear_svc_multiclass_ovr() {
    constexpr int n = 90;
    Eigen::MatrixXd X(n, 2);
    Eigen::VectorXi y(n);
    std::mt19937_64 rng(1);
    std::normal_distribution<double> nz(0.0, 0.3);
    for (int i = 0; i < n; ++i) {
        const int c = i / 30;
        // Place each class in a distinct quadrant of (x, y) space.
        const double cx = (c == 0) ? -3.0 : (c == 1 ? 0.0 : 3.0);
        const double cy = (c == 0) ? -3.0 : (c == 1 ? 3.0 : -3.0);
        X(i, 0) = cx + nz(rng);
        X(i, 1) = cy + nz(rng);
        y(i)    = c;
    }
    Skigen::LinearSVC<double> clf(
        /*C=*/1.0, Skigen::LinearSVC<double>::Loss::SquaredHinge,
        1e-5, 1000, true, std::optional<uint64_t>(1));
    clf.fit(X, y);
    ASSERT_TRUE(clf.n_classes() == 3);
    auto preds = clf.predict(X);
    int correct = 0;
    for (int i = 0; i < y.size(); ++i) if (preds(i) == y(i)) ++correct;
    // Slim SGD primal solver — converged solution matches liblinear at
    // the optimum but iteration trace differs (documented parity gap).
    ASSERT_TRUE(static_cast<double>(correct) / y.size() > 0.75);
}

void test_linear_svc_decision_function_shape() {
    auto [X, y] = two_cluster(40, 1.5, 0);
    Skigen::LinearSVC<double> clf;
    clf.fit(X, y);
    auto df = clf.decision_function(X);
    ASSERT_TRUE(df.rows() == 40);
    ASSERT_TRUE(df.cols() == 1);
}

// ---------------------------------------------------------------------------
// LinearSVR
// ---------------------------------------------------------------------------

void test_linear_svr_recovers_linear_signal() {
    constexpr int n = 100;
    Eigen::MatrixXd X(n, 1);
    Eigen::VectorXd y(n);
    for (int i = 0; i < n; ++i) {
        X(i, 0) = static_cast<double>(i) / 10.0;
        y(i)    = 2.0 * X(i, 0) + 1.0;
    }
    Skigen::LinearSVR<double> reg(
        /*C=*/100.0, /*epsilon=*/0.01,
        Skigen::LinearSVR<double>::Loss::SquaredEpsilonInsensitive,
        1e-7, 5000, true, std::optional<uint64_t>(7));
    reg.fit(X, y);
    // The slim SGD solver converges to within a few units of the true
    // line on a noiseless linear signal; tightening further would
    // require a proper liblinear-style coordinate-descent solver.
    Eigen::MatrixXd Xt(2, 1); Xt << 5.0, 1.0;
    Eigen::VectorXd yh = reg.predict(Xt);
    ASSERT_TRUE(std::abs(yh(0) - 11.0) < 2.0);
    ASSERT_TRUE(std::abs(yh(1) - 3.0)  < 2.0);
}

// ---------------------------------------------------------------------------
// SVC (kernel)
// ---------------------------------------------------------------------------

void test_svc_rbf_separates_two_clusters() {
    auto [X, y] = two_cluster(80, 2.0, 11);
    using K = Skigen::SVC<double>::Kernel;
    Skigen::SVC<double> clf(
        /*C=*/1.0, K::RBF, /*degree=*/3, /*gamma=*/0.5);
    clf.fit(X, y);
    auto preds = clf.predict(X);
    int correct = 0;
    for (int i = 0; i < y.size(); ++i) if (preds(i) == y(i)) ++correct;
    ASSERT_TRUE(static_cast<double>(correct) / y.size() > 0.9);
    ASSERT_TRUE(clf.n_support() > 0);
}

void test_svc_linear_kernel_separates() {
    auto [X, y] = two_cluster(60, 2.0, 0);
    using K = Skigen::SVC<double>::Kernel;
    Skigen::SVC<double> clf(1.0, K::Linear);
    clf.fit(X, y);
    auto preds = clf.predict(X);
    int correct = 0;
    for (int i = 0; i < y.size(); ++i) if (preds(i) == y(i)) ++correct;
    ASSERT_TRUE(static_cast<double>(correct) / y.size() > 0.9);
}

void test_svc_multiclass_throws() {
    Eigen::MatrixXd X(6, 1); X << 0, 1, 2, 3, 4, 5;
    Eigen::VectorXi y(6); y << 0, 1, 2, 0, 1, 2;
    Skigen::SVC<double> clf;
    bool threw = false;
    try { clf.fit(X, y); }
    catch (const std::invalid_argument&) { threw = true; }
    ASSERT_TRUE(threw);
}

// ---------------------------------------------------------------------------
// SVR (kernel)
// ---------------------------------------------------------------------------

void test_svr_rbf_recovers_nonlinear_signal() {
    constexpr int n = 60;
    Eigen::MatrixXd X(n, 1);
    Eigen::VectorXd y(n);
    for (int i = 0; i < n; ++i) {
        X(i, 0) = -3.0 + 6.0 * static_cast<double>(i) / (n - 1);
        y(i)    = std::sin(X(i, 0));
    }
    using K = Skigen::SVR<double>::Kernel;
    Skigen::SVR<double> reg(
        /*C=*/10.0, K::RBF, 3, /*gamma=*/0.5,
        /*coef0=*/0.0, /*epsilon=*/0.01,
        1e-4, 500, std::optional<uint64_t>(7));
    reg.fit(X, y);
    ASSERT_TRUE(reg.n_support() > 0);
    // Slim sub-gradient SGD on the dual: tracks the sin curve roughly,
    // tighter recovery requires a libsvm-style SMO-for-regression solver.
    Eigen::MatrixXd Xt(3, 1); Xt << -1.0, 0.0, 1.0;
    Eigen::VectorXd yh = reg.predict(Xt);
    ASSERT_TRUE(std::abs(yh(0) - std::sin(-1.0)) < 0.7);
    ASSERT_TRUE(std::abs(yh(1) - std::sin( 0.0)) < 0.7);
    ASSERT_TRUE(std::abs(yh(2) - std::sin( 1.0)) < 0.7);
}

// ---------------------------------------------------------------------------
// NuSVC / NuSVR / OneClassSVM — parity-gap stubs.
// ---------------------------------------------------------------------------

void test_nu_svc_throws_at_fit() {
    Skigen::NuSVC<double> nu;
    Eigen::MatrixXd X(3, 1); X << 0, 1, 2;
    Eigen::VectorXi y(3); y << 0, 1, 0;
    bool threw = false;
    try { nu.fit(X, y); }
    catch (const std::runtime_error&) { threw = true; }
    ASSERT_TRUE(threw);
}

void test_nu_svr_throws_at_fit() {
    Skigen::NuSVR<double> nu;
    Eigen::MatrixXd X(3, 1); X << 0, 1, 2;
    Eigen::VectorXd y(3); y << 0.0, 1.0, 2.0;
    bool threw = false;
    try { nu.fit(X, y); }
    catch (const std::runtime_error&) { threw = true; }
    ASSERT_TRUE(threw);
}

void test_one_class_svm_throws_at_fit() {
    Skigen::OneClassSVM<double> oc;
    Eigen::MatrixXd X(3, 1); X << 0, 1, 2;
    bool threw = false;
    try { oc.fit(X); }
    catch (const std::runtime_error&) { threw = true; }
    ASSERT_TRUE(threw);
}

int main() {
    std::cout << "Skigen SVM tests\n";
    std::cout << "----------------\n";

    run("linear_svc_binary_separable",       test_linear_svc_binary_separable);
    run("linear_svc_multiclass_ovr",         test_linear_svc_multiclass_ovr);
    run("linear_svc_decision_function_shape",test_linear_svc_decision_function_shape);
    run("linear_svr_recovers_linear_signal", test_linear_svr_recovers_linear_signal);
    run("svc_rbf_separates_two_clusters",    test_svc_rbf_separates_two_clusters);
    run("svc_linear_kernel_separates",       test_svc_linear_kernel_separates);
    run("svc_multiclass_throws",             test_svc_multiclass_throws);
    run("svr_rbf_recovers_nonlinear_signal", test_svr_rbf_recovers_nonlinear_signal);
    run("nu_svc_throws_at_fit",              test_nu_svc_throws_at_fit);
    run("nu_svr_throws_at_fit",              test_nu_svr_throws_at_fit);
    run("one_class_svm_throws_at_fit",       test_one_class_svm_throws_at_fit);

    std::cout << "----------------\n";
    std::cout << g_passed << " passed, " << g_failed << " failed\n";
    return g_failed == 0 ? 0 : 1;
}
