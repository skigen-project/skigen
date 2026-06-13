// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#include <Skigen/Dense>

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <functional>

// ---------------------------------------------------------------------------
// Minimal test harness
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
    } while (false)

#define ASSERT_THROW(expr, ExType)                                            \
    do {                                                                       \
        bool caught = false;                                                   \
        try { static_cast<void>(expr); } catch (const ExType&) { caught = true; } \
        if (!caught) {                                                         \
            throw TestFailure(std::string(__FILE__) + ":" +                    \
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

// ===================================================================
// TrainTestSplit Tests
// ===================================================================

void test_tts_basic() {
    Eigen::MatrixXd X(10, 2);
    Eigen::VectorXd y(10);
    for (int i = 0; i < 10; ++i) {
        X(i, 0) = static_cast<double>(i);
        X(i, 1) = static_cast<double>(i) * 2;
        y(i) = static_cast<double>(i);
    }

    auto result = Skigen::train_test_split(X, y, 0.3);

    // 30% of 10 = 3 test, 7 train
    ASSERT_TRUE(result.X_train.rows() == 7);
    ASSERT_TRUE(result.X_test.rows() == 3);
    ASSERT_TRUE(result.y_train.size() == 7);
    ASSERT_TRUE(result.y_test.size() == 3);
    ASSERT_TRUE(result.X_train.cols() == 2);
    ASSERT_TRUE(result.X_test.cols() == 2);
}

void test_tts_preserves_data() {
    Eigen::MatrixXd X(4, 1);
    X << 10, 20, 30, 40;
    Eigen::VectorXd y(4);
    y << 1, 2, 3, 4;

    auto result = Skigen::train_test_split(X, y, 0.25);

    // Total samples preserved
    ASSERT_TRUE(result.X_train.rows() + result.X_test.rows() == 4);
}

void test_tts_deterministic() {
    Eigen::MatrixXd X(10, 2);
    Eigen::VectorXd y(10);
    for (int i = 0; i < 10; ++i) {
        X(i, 0) = static_cast<double>(i);
        X(i, 1) = static_cast<double>(i);
        y(i) = static_cast<double>(i);
    }

    auto r1 = Skigen::train_test_split(X, y, 0.3, 42);
    auto r2 = Skigen::train_test_split(X, y, 0.3, 42);

    // Same seed → same split
    for (Eigen::Index i = 0; i < r1.y_train.size(); ++i) {
        ASSERT_NEAR(r1.y_train(i), r2.y_train(i), 1e-10);
    }
}

void test_tts_invalid_size() {
    Eigen::MatrixXd X(4, 1);
    X << 1, 2, 3, 4;
    Eigen::VectorXd y(4);
    y << 1, 2, 3, 4;

    ASSERT_THROW(Skigen::train_test_split(X, y, 0.0), std::invalid_argument);
    ASSERT_THROW(Skigen::train_test_split(X, y, 1.0), std::invalid_argument);
}

void test_tts_inconsistent_lengths() {
    Eigen::MatrixXd X(4, 1);
    X << 1, 2, 3, 4;
    Eigen::VectorXd y(3);
    y << 1, 2, 3;

    ASSERT_THROW(Skigen::train_test_split(X, y, 0.25), std::invalid_argument);
}

// ===================================================================
// ParameterGrid Tests
// ===================================================================

void test_parameter_grid_size() {
    Skigen::ParameterGrid grid(Skigen::ParameterGrid::Grid{
        {"alpha", {Skigen::ParameterValue(0.1), Skigen::ParameterValue(1.0)}},
        {"fit_intercept", {Skigen::ParameterValue(true), Skigen::ParameterValue(false)}}
    });
    ASSERT_TRUE(grid.size() == 4);
}

void test_parameter_grid_iteration() {
    Skigen::ParameterGrid grid(Skigen::ParameterGrid::Grid{
        {"alpha", {Skigen::ParameterValue(0.1), Skigen::ParameterValue(1.0)}},
    });
    int count = 0;
    for (auto params : grid) {
        (void)params;
        ++count;
    }
    ASSERT_TRUE(count == 2);
}

void test_parameter_grid_index() {
    Skigen::ParameterGrid grid(Skigen::ParameterGrid::Grid{
        {"alpha", {Skigen::ParameterValue(0.1), Skigen::ParameterValue(1.0)}},
    });
    auto p0 = grid[0];
    ASSERT_NEAR(std::get<double>(p0.at("alpha")), 0.1, 1e-10);
    auto p1 = grid[1];
    ASSERT_NEAR(std::get<double>(p1.at("alpha")), 1.0, 1e-10);
}

// ===================================================================
// GridSearchCV Tests
// ===================================================================

void test_grid_search_cv_basic() {
    Eigen::MatrixXd X(50, 2);
    Eigen::VectorXd y(50);
    for (int i = 0; i < 50; ++i) {
        X(i, 0) = static_cast<double>(i) / 50.0;
        X(i, 1) = static_cast<double>(i) / 25.0;
        y(i) = 2.0 * X(i, 0) + 0.5 * X(i, 1) + 0.1;
    }

    Skigen::Ridge<double> est;
    Skigen::ParameterGrid grid(Skigen::ParameterGrid::Grid{
        {"alpha", {Skigen::ParameterValue(0.01),
                   Skigen::ParameterValue(1.0),
                   Skigen::ParameterValue(100.0)}}
    });

    Skigen::GridSearchCV<Skigen::Ridge<double>> gs(est, grid, 3);
    gs.fit(X, y);

    ASSERT_TRUE(gs.best_score() > 0.0);
    ASSERT_TRUE(gs.cv_results_params().size() == 3);
    ASSERT_TRUE(gs.cv_results_mean_score().size() == 3);

    auto yh = gs.predict(X);
    ASSERT_TRUE(yh.size() == 50);
}

void test_grid_search_cv_best_params() {
    Eigen::MatrixXd X(60, 1);
    Eigen::VectorXd y(60);
    for (int i = 0; i < 60; ++i) {
        X(i, 0) = static_cast<double>(i);
        y(i) = 3.0 * X(i, 0) + 1.0;
    }

    Skigen::Ridge<double> est;
    Skigen::ParameterGrid grid(Skigen::ParameterGrid::Grid{
        {"alpha", {Skigen::ParameterValue(0.001),
                   Skigen::ParameterValue(1000.0)}}
    });

    Skigen::GridSearchCV<Skigen::Ridge<double>> gs(est, grid, 3);
    gs.fit(X, y);

    auto bp = gs.best_params();
    double best_alpha = std::get<double>(bp.at("alpha"));
    ASSERT_NEAR(best_alpha, 0.001, 1e-10);
}

// ===================================================================
// RandomizedSearchCV Tests
// ===================================================================

void test_randomized_search_cv_basic() {
    Eigen::MatrixXd X(50, 2);
    Eigen::VectorXd y(50);
    for (int i = 0; i < 50; ++i) {
        X(i, 0) = static_cast<double>(i) / 50.0;
        X(i, 1) = static_cast<double>(i) / 25.0;
        y(i) = 2.0 * X(i, 0) + 0.5 * X(i, 1);
    }

    Skigen::Ridge<double> est;
    Skigen::ParameterGrid dist(Skigen::ParameterGrid::Grid{
        {"alpha", {Skigen::ParameterValue(0.01),
                   Skigen::ParameterValue(0.1),
                   Skigen::ParameterValue(1.0),
                   Skigen::ParameterValue(10.0)}}
    });

    Skigen::RandomizedSearchCV<Skigen::Ridge<double>> rs(
        est, dist, 3, 3, true, 42);
    rs.fit(X, y);

    ASSERT_TRUE(rs.best_score() > 0.0);
    ASSERT_TRUE(rs.cv_results_params().size() == 3);
}

// ===================================================================
// n_jobs parallel grid dispatch + Pipeline step routing
// ===================================================================

void test_grid_search_cv_njobs_matches_serial() {
    Eigen::MatrixXd X(60, 2);
    Eigen::VectorXd y(60);
    for (int i = 0; i < 60; ++i) {
        X(i, 0) = static_cast<double>(i) * 0.1;
        X(i, 1) = std::sin(i * 0.3);
        y(i) = 2.0 * X(i, 0) - X(i, 1) + 0.5;
    }
    Skigen::ParameterGrid grid(Skigen::ParameterGrid::Grid{
        {"alpha", {Skigen::ParameterValue(0.001),
                   Skigen::ParameterValue(0.1),
                   Skigen::ParameterValue(1.0),
                   Skigen::ParameterValue(10.0)}}});

    Skigen::GridSearchCV<Skigen::Ridge<double>> serial(
        Skigen::Ridge<double>(), grid, 4, true, 1);
    Skigen::GridSearchCV<Skigen::Ridge<double>> parallel(
        Skigen::Ridge<double>(), grid, 4, true, 4);
    serial.fit(X, y);
    parallel.fit(X, y);

    // Parallel dispatch must produce identical cv scores and selection.
    ASSERT_TRUE(serial.cv_results_mean_score().size() ==
                parallel.cv_results_mean_score().size());
    for (std::size_t i = 0; i < serial.cv_results_mean_score().size(); ++i) {
        ASSERT_NEAR(serial.cv_results_mean_score()[i],
                    parallel.cv_results_mean_score()[i], 1e-12);
    }
    ASSERT_NEAR(serial.best_score(), parallel.best_score(), 1e-12);
    ASSERT_TRUE(std::get<double>(serial.best_params().at("alpha")) ==
                std::get<double>(parallel.best_params().at("alpha")));
}

void test_pipeline_set_param_index_routing() {
    auto pipe = Skigen::make_pipeline(Skigen::StandardScaler<double>(),
                                      Skigen::Ridge<double>(0.5));
    // Route "1__alpha" to the Ridge step.
    pipe.set_param("1__alpha", Skigen::ParameterValue(123.0));
    ASSERT_NEAR(pipe.template get<1>().alpha(), 123.0, 1e-12);

    // Malformed / out-of-range specs are rejected.
    ASSERT_THROW(pipe.set_param("alpha", Skigen::ParameterValue(1.0)),
                 std::invalid_argument);
    ASSERT_THROW(pipe.set_param("9__alpha", Skigen::ParameterValue(1.0)),
                 std::out_of_range);
}

int main() {
    std::cout << "=== TrainTestSplit Tests ===\n";
    run_test("tts_basic", test_tts_basic);
    run_test("tts_preserves_data", test_tts_preserves_data);
    run_test("tts_deterministic", test_tts_deterministic);
    run_test("tts_invalid_size", test_tts_invalid_size);
    run_test("tts_inconsistent_lengths", test_tts_inconsistent_lengths);

    std::cout << "\n=== ParameterGrid Tests ===\n";
    run_test("parameter_grid_size", test_parameter_grid_size);
    run_test("parameter_grid_iteration", test_parameter_grid_iteration);
    run_test("parameter_grid_index", test_parameter_grid_index);

    std::cout << "\n=== GridSearchCV Tests ===\n";
    run_test("grid_search_cv_basic", test_grid_search_cv_basic);
    run_test("grid_search_cv_best_params", test_grid_search_cv_best_params);

    run_test("grid_search_cv_njobs_matches_serial",
             test_grid_search_cv_njobs_matches_serial);
    run_test("pipeline_set_param_index_routing",
             test_pipeline_set_param_index_routing);

    std::cout << "\n=== RandomizedSearchCV Tests ===\n";
    run_test("randomized_search_cv_basic", test_randomized_search_cv_basic);

    std::cout << "\n" << g_passed << " passed, " << g_failed << " failed.\n";
    return g_failed > 0 ? 1 : 0;
}
