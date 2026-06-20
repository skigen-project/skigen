// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#include <Skigen/NeuralNetwork>

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
// MLPRegressor (SGD)
// ---------------------------------------------------------------------------

void test_mlp_regressor_recovers_linear_signal() {
    constexpr int n = 200;
    Eigen::MatrixXd X(n, 1);
    Eigen::VectorXd y(n);
    for (int i = 0; i < n; ++i) {
        X(i, 0) = -1.0 + 2.0 * static_cast<double>(i) / (n - 1);
        y(i)    = 2.0 * X(i, 0) + 0.5;
    }
    Skigen::MLPRegressor<double> mlp(
        /*hidden_layer_sizes=*/{16},
        Skigen::MLPActivation::ReLU,
        Skigen::MLPSolver::SGD,
        /*alpha=*/1e-5, /*lr=*/0.05, /*max_iter=*/500,
        /*tol=*/1e-7, /*batch_size=*/0,
        std::optional<uint64_t>(7));
    mlp.fit(X, y);
    Eigen::MatrixXd Xt(2, 1); Xt << 0.5, -0.5;
    Eigen::VectorXd yh = mlp.predict(Xt);
    ASSERT_TRUE(std::abs(yh(0) - 1.5)  < 0.4);
    ASSERT_TRUE(std::abs(yh(1) - -0.5) < 0.4);
    ASSERT_TRUE(mlp.coefs().size() == 2);
    ASSERT_TRUE(mlp.intercepts().size() == 2);
}

void test_mlp_regressor_n_iter_and_loss_recorded() {
    Eigen::MatrixXd X(50, 1);
    Eigen::VectorXd y(50);
    for (int i = 0; i < 50; ++i) {
        X(i, 0) = i;
        y(i)    = 2.0 * X(i, 0) + 1.0;
    }
    Skigen::MLPRegressor<double> mlp(
        {8}, Skigen::MLPActivation::Tanh,
        Skigen::MLPSolver::SGD,
        1e-4, 1e-2, /*max_iter=*/30, 1e-9, 0,
        std::optional<uint64_t>(0));
    mlp.fit(X, y);
    ASSERT_TRUE(mlp.n_iter_run() > 0);
    ASSERT_TRUE(mlp.n_iter_run() <= 30);
    ASSERT_TRUE(mlp.loss() > 0.0);
}

void test_mlp_regressor_multi_layer() {
    Eigen::MatrixXd X(80, 1);
    Eigen::VectorXd y(80);
    for (int i = 0; i < 80; ++i) {
        X(i, 0) = i / 10.0;
        y(i)    = X(i, 0);
    }
    Skigen::MLPRegressor<double> mlp(
        {8, 8}, Skigen::MLPActivation::ReLU,
        Skigen::MLPSolver::SGD,
        1e-5, 0.05, 300, 1e-8, 0,
        std::optional<uint64_t>(7));
    mlp.fit(X, y);
    ASSERT_TRUE(mlp.coefs().size() == 3);
}

// ---------------------------------------------------------------------------
// MLPRegressor (Adam)
// ---------------------------------------------------------------------------

void test_mlp_regressor_adam_converges() {
    constexpr int n = 200;
    Eigen::MatrixXd X(n, 1);
    Eigen::VectorXd y(n);
    for (int i = 0; i < n; ++i) {
        X(i, 0) = -1.0 + 2.0 * static_cast<double>(i) / (n - 1);
        y(i)    = 2.0 * X(i, 0) + 0.5;
    }
    Skigen::MLPRegressor<double> mlp(
        {16}, Skigen::MLPActivation::ReLU,
        Skigen::MLPSolver::Adam,
        /*alpha=*/1e-5, /*lr=*/1e-3, /*max_iter=*/500,
        /*tol=*/1e-7, /*batch_size=*/0,
        std::optional<uint64_t>(7));
    mlp.fit(X, y);
    double r2 = mlp.score(X, y);
    ASSERT_TRUE(r2 > 0.9);
}

void test_mlp_regressor_adam_is_default_solver() {
    Skigen::MLPRegressor<double> mlp;
    ASSERT_TRUE(mlp.solver() == Skigen::MLPSolver::Adam);
}

// ---------------------------------------------------------------------------
// MLPRegressor multi-target
// ---------------------------------------------------------------------------

void test_mlp_regressor_multi_target() {
    constexpr int n = 200;
    Eigen::MatrixXd X(n, 2);
    Eigen::MatrixXd Y(n, 2);
    for (int i = 0; i < n; ++i) {
        X(i, 0) = -1.0 + 2.0 * static_cast<double>(i) / (n - 1);
        X(i, 1) = 0.5 * X(i, 0);
        Y(i, 0) = 2.0 * X(i, 0) + 0.5;
        Y(i, 1) = -X(i, 1) + 1.0;
    }
    Skigen::MLPRegressor<double> mlp(
        {32}, Skigen::MLPActivation::ReLU,
        Skigen::MLPSolver::Adam,
        1e-5, 1e-3, 500, 1e-7, 0,
        std::optional<uint64_t>(42));
    mlp.fit_multi(X, Y);
    ASSERT_TRUE(mlp.n_outputs() == 2);
    Eigen::MatrixXd Yh = mlp.predict_multi(X);
    ASSERT_TRUE(Yh.cols() == 2);
    ASSERT_TRUE(Yh.rows() == n);
}

// ---------------------------------------------------------------------------
// MLPClassifier (SGD, binary)
// ---------------------------------------------------------------------------

void test_mlp_classifier_separates_two_clusters() {
    constexpr int n = 200;
    Eigen::MatrixXd X(n, 2);
    Eigen::VectorXi y(n);
    std::mt19937_64 rng(7);
    std::normal_distribution<double> nz(0.0, 0.5);
    for (int i = 0; i < n; ++i) {
        const double cls = (i < n / 2) ? -1.5 : 1.5;
        X(i, 0) = cls + nz(rng);
        X(i, 1) = cls + nz(rng);
        y(i)    = (cls > 0) ? 1 : 0;
    }
    Skigen::MLPClassifier<double> mlp(
        {16}, Skigen::MLPActivation::ReLU,
        Skigen::MLPSolver::SGD,
        /*alpha=*/1e-4, /*lr=*/0.05, /*max_iter=*/300,
        /*tol=*/1e-6, /*batch_size=*/0,
        std::optional<uint64_t>(7));
    mlp.fit(X, y);
    auto preds = mlp.predict(X);
    int correct = 0;
    for (int i = 0; i < y.size(); ++i) if (preds(i) == y(i)) ++correct;
    ASSERT_TRUE(static_cast<double>(correct) / y.size() > 0.9);
}

void test_mlp_classifier_predict_proba_rows_sum_to_one() {
    Eigen::MatrixXd X(30, 2);
    Eigen::VectorXi y(30);
    for (int i = 0; i < 30; ++i) {
        X(i, 0) = i / 10.0;
        X(i, 1) = i % 5;
        y(i)    = (i >= 15) ? 1 : 0;
    }
    Skigen::MLPClassifier<double> mlp(
        {8}, Skigen::MLPActivation::Tanh,
        Skigen::MLPSolver::SGD,
        1e-4, 0.05, 200, 1e-7, 0,
        std::optional<uint64_t>(0));
    mlp.fit(X, y);
    Eigen::MatrixXd P = mlp.predict_proba(X);
    ASSERT_TRUE(P.rows() == 30);
    ASSERT_TRUE(P.cols() == 2);
    for (int i = 0; i < 30; ++i) {
        ASSERT_NEAR(P(i, 0) + P(i, 1), 1.0, 1e-12);
        ASSERT_TRUE(P(i, 0) >= 0.0 && P(i, 0) <= 1.0);
    }
}

// ---------------------------------------------------------------------------
// MLPClassifier (Adam, binary)
// ---------------------------------------------------------------------------

void test_mlp_classifier_adam_binary() {
    constexpr int n = 200;
    Eigen::MatrixXd X(n, 2);
    Eigen::VectorXi y(n);
    std::mt19937_64 rng(42);
    std::normal_distribution<double> nz(0.0, 0.4);
    for (int i = 0; i < n; ++i) {
        const double cls = (i < n / 2) ? -1.5 : 1.5;
        X(i, 0) = cls + nz(rng);
        X(i, 1) = cls + nz(rng);
        y(i)    = (cls > 0) ? 1 : 0;
    }
    Skigen::MLPClassifier<double> mlp(
        {16}, Skigen::MLPActivation::ReLU,
        Skigen::MLPSolver::Adam,
        1e-4, 1e-3, 300, 1e-7, 0,
        std::optional<uint64_t>(42));
    mlp.fit(X, y);
    auto preds = mlp.predict(X);
    int correct = 0;
    for (int i = 0; i < y.size(); ++i) if (preds(i) == y(i)) ++correct;
    ASSERT_TRUE(static_cast<double>(correct) / y.size() > 0.9);
}

// ---------------------------------------------------------------------------
// MLPClassifier (multiclass)
// ---------------------------------------------------------------------------

void test_mlp_classifier_multiclass_three_classes() {
    constexpr int n = 300;
    Eigen::MatrixXd X(n, 2);
    Eigen::VectorXi y(n);
    std::mt19937_64 rng(7);
    std::normal_distribution<double> nz(0.0, 0.3);
    for (int i = 0; i < n; ++i) {
        const int cls = i % 3;
        y(i) = cls;
        double cx = 0, cy = 0;
        if (cls == 0) { cx = -2.0; cy = 0.0; }
        else if (cls == 1) { cx = 2.0; cy = 0.0; }
        else { cx = 0.0; cy = 2.0; }
        X(i, 0) = cx + nz(rng);
        X(i, 1) = cy + nz(rng);
    }
    Skigen::MLPClassifier<double> mlp(
        {32}, Skigen::MLPActivation::ReLU,
        Skigen::MLPSolver::Adam,
        1e-4, 1e-3, 500, 1e-7, 0,
        std::optional<uint64_t>(7));
    mlp.fit(X, y);
    ASSERT_TRUE(mlp.n_classes() == 3);
    auto preds = mlp.predict(X);
    int correct = 0;
    for (int i = 0; i < n; ++i) if (preds(i) == y(i)) ++correct;
    ASSERT_TRUE(static_cast<double>(correct) / n > 0.85);
}

void test_mlp_classifier_multiclass_predict_proba_sums_to_one() {
    constexpr int n = 120;
    Eigen::MatrixXd X(n, 2);
    Eigen::VectorXi y(n);
    std::mt19937_64 rng(42);
    std::normal_distribution<double> nz(0.0, 0.3);
    for (int i = 0; i < n; ++i) {
        const int cls = i % 4;
        y(i) = cls;
        X(i, 0) = (cls < 2 ? -2.0 : 2.0) + nz(rng);
        X(i, 1) = (cls % 2 == 0 ? -2.0 : 2.0) + nz(rng);
    }
    Skigen::MLPClassifier<double> mlp(
        {16}, Skigen::MLPActivation::ReLU,
        Skigen::MLPSolver::Adam,
        1e-4, 1e-3, 300, 1e-7, 0,
        std::optional<uint64_t>(42));
    mlp.fit(X, y);
    Eigen::MatrixXd P = mlp.predict_proba(X);
    ASSERT_TRUE(P.cols() == 4);
    ASSERT_TRUE(P.rows() == n);
    for (int i = 0; i < n; ++i) {
        ASSERT_NEAR(P.row(i).sum(), 1.0, 1e-10);
        for (int c = 0; c < 4; ++c)
            ASSERT_TRUE(P(i, c) >= 0.0 && P(i, c) <= 1.0);
    }
}

void test_mlp_classifier_lbfgs_throws() {
    bool threw = false;
    try {
        Skigen::MLPClassifier<double> mlp(
            {8}, Skigen::MLPActivation::ReLU,
            Skigen::MLPSolver::LBFGS);
    } catch (const std::invalid_argument&) { threw = true; }
    ASSERT_TRUE(threw);
}

void test_mlp_regressor_lbfgs_throws() {
    bool threw = false;
    try {
        Skigen::MLPRegressor<double> mlp(
            {8}, Skigen::MLPActivation::ReLU,
            Skigen::MLPSolver::LBFGS);
    } catch (const std::invalid_argument&) { threw = true; }
    ASSERT_TRUE(threw);
}

int main() {
    std::cout << "Skigen NeuralNetwork tests\n";
    std::cout << "--------------------------\n";

    run("mlp_regressor_recovers_linear_signal",
        test_mlp_regressor_recovers_linear_signal);
    run("mlp_regressor_n_iter_and_loss_recorded",
        test_mlp_regressor_n_iter_and_loss_recorded);
    run("mlp_regressor_multi_layer",
        test_mlp_regressor_multi_layer);
    run("mlp_regressor_adam_converges",
        test_mlp_regressor_adam_converges);
    run("mlp_regressor_adam_is_default_solver",
        test_mlp_regressor_adam_is_default_solver);
    run("mlp_regressor_multi_target",
        test_mlp_regressor_multi_target);
    run("mlp_classifier_separates_two_clusters",
        test_mlp_classifier_separates_two_clusters);
    run("mlp_classifier_predict_proba_rows_sum_to_one",
        test_mlp_classifier_predict_proba_rows_sum_to_one);
    run("mlp_classifier_adam_binary",
        test_mlp_classifier_adam_binary);
    run("mlp_classifier_multiclass_three_classes",
        test_mlp_classifier_multiclass_three_classes);
    run("mlp_classifier_multiclass_predict_proba_sums_to_one",
        test_mlp_classifier_multiclass_predict_proba_sums_to_one);
    run("mlp_classifier_lbfgs_throws",
        test_mlp_classifier_lbfgs_throws);
    run("mlp_regressor_lbfgs_throws",
        test_mlp_regressor_lbfgs_throws);

    std::cout << "--------------------------\n";
    std::cout << g_passed << " passed, " << g_failed << " failed\n";
    return g_failed == 0 ? 0 : 1;
}
