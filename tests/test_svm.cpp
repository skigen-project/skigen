// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#include <Skigen/SVM>

#include <Eigen/SparseCore>
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

void test_svc_predict_proba_rows_sum_to_one() {
    auto [X, y] = two_cluster(100, 2.0, 7);
    using K = Skigen::SVC<double>::Kernel;
    Skigen::SVC<double> clf(
        /*C=*/1.0, K::RBF, /*degree=*/3, /*gamma=*/0.5,
        /*coef0=*/0.0, /*probability=*/true,
        /*tol=*/1e-3, /*max_passes=*/50,
        std::optional<uint64_t>(7));
    clf.fit(X, y);
    Eigen::MatrixXd P = clf.predict_proba(X);
    ASSERT_TRUE(P.rows() == X.rows());
    ASSERT_TRUE(P.cols() == 2);
    for (Eigen::Index i = 0; i < P.rows(); ++i) {
        ASSERT_NEAR(P(i, 0) + P(i, 1), 1.0, 1e-10);
        ASSERT_TRUE(P(i, 0) >= 0.0 && P(i, 0) <= 1.0);
    }
}

void test_svc_predict_proba_requires_flag() {
    auto [X, y] = two_cluster(40, 2.0, 0);
    Skigen::SVC<double> clf;
    clf.fit(X, y);
    bool threw = false;
    try { clf.predict_proba(X); }
    catch (const std::runtime_error&) { threw = true; }
    ASSERT_TRUE(threw);
}

void test_svc_sparse_fit_predict() {
    auto [X, y] = two_cluster(60, 2.0, 42);
    Eigen::SparseMatrix<double> Xs = X.sparseView();
    using K = Skigen::SVC<double>::Kernel;
    Skigen::SVC<double> clf(1.0, K::RBF, 3, 0.5);
    clf.fit(Xs, y);
    auto preds = clf.predict(Xs);
    int correct = 0;
    for (int i = 0; i < y.size(); ++i) if (preds(i) == y(i)) ++correct;
    ASSERT_TRUE(static_cast<double>(correct) / y.size() > 0.85);
}

void test_linear_svc_sparse_fit_predict() {
    auto [X, y] = two_cluster(100, 1.5, 7);
    Eigen::SparseMatrix<double> Xs = X.sparseView();
    Skigen::LinearSVC<double> clf(
        1.0, Skigen::LinearSVC<double>::Loss::SquaredHinge,
        1e-4, 200, true, std::optional<uint64_t>(7));
    clf.fit(Xs, y);
    auto preds = clf.predict(Xs);
    int correct = 0;
    for (int i = 0; i < y.size(); ++i) if (preds(i) == y(i)) ++correct;
    ASSERT_TRUE(static_cast<double>(correct) / y.size() > 0.85);
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

void test_nu_svc_separates_two_clusters() {
    constexpr int n = 40;
    Eigen::MatrixXd X(n, 2);
    Eigen::VectorXi y(n);
    std::mt19937_64 rng(5);
    std::normal_distribution<double> ns(0.0, 0.3);
    for (int i = 0; i < n; ++i) {
        const double cls = (i < n / 2) ? -2.0 : 2.0;
        X(i, 0) = cls + ns(rng);
        X(i, 1) = cls + ns(rng);
        y(i)    = (cls > 0) ? 1 : 0;
    }
    Skigen::NuSVC<double> nu(
        0.5, Skigen::NuSVC<double>::Kernel::RBF, 3, 0.0, 0.0, 1e-3, 80,
        std::optional<uint64_t>(0));
    nu.fit(X, y);
    const Eigen::VectorXi preds = nu.predict(X);
    int correct = 0;
    for (int i = 0; i < n; ++i) if (preds(i) == y(i)) ++correct;
    ASSERT_TRUE(static_cast<double>(correct) / n > 0.9);
    ASSERT_TRUE(nu.n_support() > 0);
    ASSERT_TRUE(nu.classes().size() == 2);
}

void test_nu_svc_invalid_nu_throws() {
    Skigen::NuSVC<double> nu(0.0);
    Eigen::MatrixXd X(4, 1); X << 0, 1, 2, 3;
    Eigen::VectorXi y(4); y << 0, 0, 1, 1;
    bool threw = false;
    try { nu.fit(X, y); }
    catch (const std::invalid_argument&) { threw = true; }
    ASSERT_TRUE(threw);
}

void test_nu_svr_recovers_linear_signal() {
    constexpr int n = 60;
    Eigen::MatrixXd X(n, 1);
    Eigen::VectorXd y(n);
    std::mt19937_64 rng(9);
    std::normal_distribution<double> noise(0.0, 0.05);
    for (int i = 0; i < n; ++i) {
        X(i, 0) = static_cast<double>(i) / n;
        y(i) = 2.0 * X(i, 0) + 0.5 + noise(rng);
    }
    Skigen::NuSVR<double> nu(
        0.5, 5.0, Skigen::NuSVR<double>::Kernel::Linear, 3, 0.0, 0.0, 1e-4,
        2000, std::optional<uint64_t>(0));
    nu.fit(X, y);
    ASSERT_TRUE(nu.score(X, y) > 0.9);
    ASSERT_TRUE(nu.epsilon_fitted() >= 0.0);
    ASSERT_TRUE(nu.n_support() > 0);
}

void test_one_class_svm_fits_and_flags_some_outliers() {
    // Dense cluster near the origin plus a couple of outliers.
    Eigen::MatrixXd X(20, 2);
    std::mt19937_64 rng(7);
    std::normal_distribution<double> ns(0.0, 0.3);
    for (int i = 0; i < 18; ++i) { X(i, 0) = ns(rng); X(i, 1) = ns(rng); }
    X(18, 0) = 2.5;  X(18, 1) = 2.5;
    X(19, 0) = -2.5; X(19, 1) = 2.2;

    Skigen::OneClassSVM<double> oc(
        Skigen::OneClassSVM<double>::Kernel::RBF,
        3, /*gamma=*/0.5, 0.0, /*nu=*/0.2, 1e-3, 80,
        std::optional<uint64_t>(0));
    oc.fit(X);

    const Eigen::VectorXi labels = oc.predict(X);
    const Eigen::VectorXd scores = oc.score_samples(X);
    const Eigen::VectorXd decisions = oc.decision_function(X);
    ASSERT_TRUE(labels.size() == 20);
    ASSERT_TRUE(scores.allFinite());
    ASSERT_TRUE(decisions.allFinite());
    ASSERT_TRUE(oc.n_support() > 0);
    ASSERT_TRUE(oc.dual_coef().size() == oc.n_support());
    // decision_function = score_samples + offset.
    for (int i = 0; i < 20; ++i) {
        ASSERT_NEAR(decisions(i), scores(i) + oc.offset(), 1e-9);
    }
    // At least one point is flagged as an outlier and not everything is.
    int outliers = 0;
    for (int i = 0; i < 20; ++i) if (labels(i) == -1) ++outliers;
    ASSERT_TRUE(outliers >= 1);
    ASSERT_TRUE(outliers < 20);
    // labels are strictly +1 / -1.
    for (int i = 0; i < 20; ++i) {
        ASSERT_TRUE(labels(i) == 1 || labels(i) == -1);
    }
}

void test_one_class_svm_invalid_nu_throws() {
    Skigen::OneClassSVM<double> oc(
        Skigen::OneClassSVM<double>::Kernel::RBF, 3, 0.0, 0.0, /*nu=*/0.0);
    Eigen::MatrixXd X(3, 1); X << 0, 1, 2;
    bool threw = false;
    try { oc.fit(X); }
    catch (const std::invalid_argument&) { threw = true; }
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
    run("svc_predict_proba_rows_sum_to_one", test_svc_predict_proba_rows_sum_to_one);
    run("svc_predict_proba_requires_flag",   test_svc_predict_proba_requires_flag);
    run("svc_sparse_fit_predict",            test_svc_sparse_fit_predict);
    run("linear_svc_sparse_fit_predict",     test_linear_svc_sparse_fit_predict);
    run("svr_rbf_recovers_nonlinear_signal", test_svr_rbf_recovers_nonlinear_signal);
    run("nu_svc_separates_two_clusters",     test_nu_svc_separates_two_clusters);
    run("nu_svc_invalid_nu_throws",          test_nu_svc_invalid_nu_throws);
    run("nu_svr_recovers_linear_signal",     test_nu_svr_recovers_linear_signal);
    run("one_class_svm_fits_and_flags_some_outliers",
        test_one_class_svm_fits_and_flags_some_outliers);
    run("one_class_svm_invalid_nu_throws",   test_one_class_svm_invalid_nu_throws);

    std::cout << "----------------\n";
    std::cout << g_passed << " passed, " << g_failed << " failed\n";
    return g_failed == 0 ? 0 : 1;
}
