// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#include <Skigen/Calibration>
#include <Skigen/NaiveBayes>

#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

// ---------------------------------------------------------------------------
// Minimal test harness (mirrors test_standard_scaler.cpp)
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
// Helpers — small binary-classification dataset.
// ---------------------------------------------------------------------------

static std::pair<Eigen::MatrixXd, Eigen::VectorXi>
make_two_cluster_dataset(int n, double sep, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> nz(0.0, 1.0);
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
// CalibratedClassifierCV (sigmoid)
// ---------------------------------------------------------------------------

void test_calibrated_sigmoid_runs_and_predicts_correctly() {
    auto [X, y] = make_two_cluster_dataset(200, 1.5, 42);
    Skigen::GaussianNB<double> nb;
    Skigen::CalibratedClassifierCV<Skigen::GaussianNB<double>, double> cc(
        nb, Skigen::CalibrationMethod::Sigmoid, /*cv=*/5,
        /*n_jobs=*/1, /*ensemble=*/true,
        std::optional<uint64_t>(7));
    cc.fit(X, y);

    auto preds = cc.predict(X);
    int correct = 0;
    for (int i = 0; i < y.size(); ++i) if (preds(i) == y(i)) ++correct;
    const double acc = static_cast<double>(correct) / y.size();
    ASSERT_TRUE(acc > 0.85);
    ASSERT_TRUE(cc.n_estimators_fitted() == 5);
    ASSERT_TRUE(cc.n_classes() == 2);
}

void test_calibrated_sigmoid_predict_proba_rows_sum_to_one() {
    auto [X, y] = make_two_cluster_dataset(120, 1.2, 11);
    Skigen::GaussianNB<double> nb;
    Skigen::CalibratedClassifierCV<Skigen::GaussianNB<double>, double> cc(
        nb, Skigen::CalibrationMethod::Sigmoid, 5, 1, true,
        std::optional<uint64_t>(11));
    cc.fit(X, y);
    Eigen::MatrixXd P = cc.predict_proba(X);
    ASSERT_TRUE(P.rows() == X.rows());
    ASSERT_TRUE(P.cols() == 2);
    for (int i = 0; i < P.rows(); ++i) {
        ASSERT_NEAR(P(i, 0) + P(i, 1), 1.0, 1e-12);
        ASSERT_TRUE(P(i, 0) >= 0.0 && P(i, 0) <= 1.0);
        ASSERT_TRUE(P(i, 1) >= 0.0 && P(i, 1) <= 1.0);
    }
}

// ---------------------------------------------------------------------------
// CalibratedClassifierCV (isotonic)
// ---------------------------------------------------------------------------

void test_calibrated_isotonic_runs_and_predicts_correctly() {
    auto [X, y] = make_two_cluster_dataset(200, 1.5, 42);
    Skigen::GaussianNB<double> nb;
    Skigen::CalibratedClassifierCV<Skigen::GaussianNB<double>, double> cc(
        nb, Skigen::CalibrationMethod::Isotonic, 5, 1, true,
        std::optional<uint64_t>(7));
    cc.fit(X, y);
    auto preds = cc.predict(X);
    int correct = 0;
    for (int i = 0; i < y.size(); ++i) if (preds(i) == y(i)) ++correct;
    const double acc = static_cast<double>(correct) / y.size();
    ASSERT_TRUE(acc > 0.85);
}

void test_calibrated_isotonic_predict_proba_in_range() {
    auto [X, y] = make_two_cluster_dataset(150, 1.2, 22);
    Skigen::GaussianNB<double> nb;
    Skigen::CalibratedClassifierCV<Skigen::GaussianNB<double>, double> cc(
        nb, Skigen::CalibrationMethod::Isotonic, 5, 1, true,
        std::optional<uint64_t>(22));
    cc.fit(X, y);
    Eigen::MatrixXd P = cc.predict_proba(X);
    for (int i = 0; i < P.rows(); ++i) {
        ASSERT_NEAR(P(i, 0) + P(i, 1), 1.0, 1e-12);
        ASSERT_TRUE(P(i, 1) >= 0.0 && P(i, 1) <= 1.0);
    }
}

// ---------------------------------------------------------------------------
// Reproducibility, edge cases, errors
// ---------------------------------------------------------------------------

void test_calibrated_random_state_reproducible() {
    auto [X, y] = make_two_cluster_dataset(120, 1.2, 99);
    Skigen::GaussianNB<double> nb;

    Skigen::CalibratedClassifierCV<Skigen::GaussianNB<double>, double> a(
        nb, Skigen::CalibrationMethod::Sigmoid, 5, 1, true,
        std::optional<uint64_t>(13));
    Skigen::CalibratedClassifierCV<Skigen::GaussianNB<double>, double> b(
        nb, Skigen::CalibrationMethod::Sigmoid, 5, 1, true,
        std::optional<uint64_t>(13));
    a.fit(X, y);
    b.fit(X, y);

    auto pa = a.predict(X);
    auto pb = b.predict(X);
    for (int i = 0; i < pa.size(); ++i) ASSERT_TRUE(pa(i) == pb(i));
}

void test_calibrated_works_with_bernoulli_nb() {
    // BernoulliNB is copy-constructible, mirroring the GaussianNB path on
    // a sparser binarised feature set.
    auto [X, y] = make_two_cluster_dataset(120, 1.2, 5);
    // Binarise X around 0.
    Eigen::MatrixXd Xb(X.rows(), X.cols());
    for (Eigen::Index i = 0; i < X.rows(); ++i)
        for (Eigen::Index j = 0; j < X.cols(); ++j)
            Xb(i, j) = X(i, j) > 0.0 ? 1.0 : 0.0;

    Skigen::BernoulliNB<double> bnb;
    Skigen::CalibratedClassifierCV<Skigen::BernoulliNB<double>, double> cc(
        bnb, Skigen::CalibrationMethod::Isotonic, 3, 1, true,
        std::optional<uint64_t>(5));
    cc.fit(Xb, y);
    auto preds = cc.predict(Xb);
    int correct = 0;
    for (int i = 0; i < y.size(); ++i) if (preds(i) == y(i)) ++correct;
    const double acc = static_cast<double>(correct) / y.size();
    ASSERT_TRUE(acc > 0.7);
}

void test_calibrated_multiclass_throws() {
    Eigen::MatrixXd X(9, 1);
    Eigen::VectorXi y(9);
    for (int i = 0; i < 9; ++i) { X(i, 0) = i; y(i) = i % 3; }
    Skigen::GaussianNB<double> nb;
    Skigen::CalibratedClassifierCV<Skigen::GaussianNB<double>, double> cc(
        nb, Skigen::CalibrationMethod::Sigmoid, 3);
    bool threw = false;
    try { cc.fit(X, y); }
    catch (const std::invalid_argument&) { threw = true; }
    ASSERT_TRUE(threw);
}

void test_calibrated_cv_too_small_throws() {
    Skigen::GaussianNB<double> nb;
    bool threw = false;
    try {
        Skigen::CalibratedClassifierCV<Skigen::GaussianNB<double>, double> cc(
            nb, Skigen::CalibrationMethod::Sigmoid, /*cv=*/1);
        (void)cc;
    } catch (const std::invalid_argument&) { threw = true; }
    ASSERT_TRUE(threw);
}

void test_calibrated_sigmoid_ensemble_false_runs() {
    auto [X, y] = make_two_cluster_dataset(160, 1.4, 88);
    Skigen::GaussianNB<double> nb;
    Skigen::CalibratedClassifierCV<Skigen::GaussianNB<double>, double> cc(
        nb, Skigen::CalibrationMethod::Sigmoid, 4, 1,
        /*ensemble=*/false, std::optional<uint64_t>(88));
    cc.fit(X, y);
    ASSERT_TRUE(cc.n_estimators_fitted() == 1);
    Eigen::MatrixXd P = cc.predict_proba(X);
    ASSERT_TRUE(P.rows() == X.rows());
    ASSERT_TRUE(P.cols() == 2);
    for (int i = 0; i < P.rows(); ++i) {
        ASSERT_NEAR(P(i, 0) + P(i, 1), 1.0, 1e-12);
    }
}

void test_calibrated_isotonic_ensemble_false_runs() {
    auto [X, y] = make_two_cluster_dataset(160, 1.4, 89);
    Skigen::GaussianNB<double> nb;
    Skigen::CalibratedClassifierCV<Skigen::GaussianNB<double>, double> cc(
        nb, Skigen::CalibrationMethod::Isotonic, 4, 1,
        /*ensemble=*/false, std::optional<uint64_t>(89));
    cc.fit(X, y);
    ASSERT_TRUE(cc.n_estimators_fitted() == 1);
    Eigen::MatrixXd P = cc.predict_proba(X);
    for (int i = 0; i < P.rows(); ++i) {
        ASSERT_NEAR(P(i, 0) + P(i, 1), 1.0, 1e-12);
        ASSERT_TRUE(P(i, 1) >= 0.0 && P(i, 1) <= 1.0);
    }
}

void test_calibrated_not_fitted_throws() {
    Skigen::GaussianNB<double> nb;
    Skigen::CalibratedClassifierCV<Skigen::GaussianNB<double>, double> cc(nb);
    Eigen::MatrixXd X(2, 1); X << 0.0, 1.0;
    bool threw = false;
    try { (void)cc.predict_proba(X); }
    catch (const std::runtime_error&) { threw = true; }
    ASSERT_TRUE(threw);
}

int main() {
    std::cout << "Skigen CalibratedClassifierCV unit tests\n";
    std::cout << "----------------------------------------\n";
    run("sigmoid_runs_and_predicts_correctly",
        test_calibrated_sigmoid_runs_and_predicts_correctly);
    run("sigmoid_predict_proba_rows_sum_to_one",
        test_calibrated_sigmoid_predict_proba_rows_sum_to_one);
    run("isotonic_runs_and_predicts_correctly",
        test_calibrated_isotonic_runs_and_predicts_correctly);
    run("isotonic_predict_proba_in_range",
        test_calibrated_isotonic_predict_proba_in_range);
    run("random_state_reproducible",
        test_calibrated_random_state_reproducible);
    run("works_with_bernoulli_nb",
        test_calibrated_works_with_bernoulli_nb);
    run("multiclass_throws",
        test_calibrated_multiclass_throws);
    run("cv_too_small_throws",
        test_calibrated_cv_too_small_throws);
    run("sigmoid_ensemble_false_runs",
        test_calibrated_sigmoid_ensemble_false_runs);
    run("isotonic_ensemble_false_runs",
        test_calibrated_isotonic_ensemble_false_runs);
    run("not_fitted_throws",
        test_calibrated_not_fitted_throws);

    std::cout << "----------------------------------------\n";
    std::cout << g_passed << " passed, " << g_failed << " failed\n";
    return g_failed == 0 ? 0 : 1;
}
