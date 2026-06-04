// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#include <Skigen/Dense>

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <functional>
#include <vector>

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

template <typename MatA, typename MatB, typename Scalar>
void assert_matrix_near(const MatA& A, const MatB& B, Scalar tol,
                        const char* file, int line) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        std::ostringstream oss;
        oss << file << ":" << line << ": Shape mismatch";
        throw TestFailure(oss.str());
    }
    for (Eigen::Index i = 0; i < A.rows(); ++i)
        for (Eigen::Index j = 0; j < A.cols(); ++j)
            assert_near(A(i, j), B(i, j), tol, file, line);
}
#define ASSERT_MATRIX_NEAR(A, B, tol) assert_matrix_near(A, B, tol, __FILE__, __LINE__)

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
// MinMaxScaler Tests
// ===================================================================

void test_minmax_basic() {
    Eigen::MatrixXd X(4, 2);
    X << 1, 10,
         2, 20,
         3, 30,
         4, 40;

    Skigen::MinMaxScaler scaler;
    scaler.fit(X);

    Eigen::MatrixXd Z = scaler.transform(X);
    // min should be 0, max should be 1
    ASSERT_NEAR(Z.col(0).minCoeff(), 0.0, 1e-12);
    ASSERT_NEAR(Z.col(0).maxCoeff(), 1.0, 1e-12);
    ASSERT_NEAR(Z.col(1).minCoeff(), 0.0, 1e-12);
    ASSERT_NEAR(Z.col(1).maxCoeff(), 1.0, 1e-12);
}

void test_minmax_round_trip() {
    Eigen::MatrixXd X(4, 3);
    X << 1, 10, 100,
         2, 20, 200,
         3, 30, 300,
         4, 40, 400;

    Skigen::MinMaxScaler scaler;
    scaler.fit(X);
    Eigen::MatrixXd Z = scaler.transform(X);
    Eigen::MatrixXd X_back = scaler.inverse_transform(Z);
    ASSERT_MATRIX_NEAR(X, X_back, 1e-12);
}

void test_minmax_custom_range() {
    Eigen::MatrixXd X(3, 1);
    X << 0, 5, 10;

    Skigen::MinMaxScaler<double> scaler({-1.0, 1.0});
    scaler.fit(X);
    Eigen::MatrixXd Z = scaler.transform(X);

    ASSERT_NEAR(Z(0, 0), -1.0, 1e-12);
    ASSERT_NEAR(Z(1, 0),  0.0, 1e-12);
    ASSERT_NEAR(Z(2, 0),  1.0, 1e-12);
}

void test_minmax_constant_feature() {
    Eigen::MatrixXd X(3, 2);
    X << 1, 5,
         2, 5,
         3, 5;

    Skigen::MinMaxScaler scaler;
    scaler.fit(X);
    Eigen::MatrixXd Z = scaler.transform(X);
    // Constant column: data_range=0, scale handled, result should be min of feature_range
    ASSERT_TRUE(std::isfinite(Z(0, 1)));
}

void test_minmax_inplace() {
    Eigen::MatrixXd X(3, 2);
    X << 1, 2, 3, 4, 5, 6;

    Skigen::MinMaxScaler scaler;
    scaler.fit(X);
    Eigen::MatrixXd Z = scaler.transform(X);
    Eigen::MatrixXd X_mut = X;
    scaler.transform_inplace(X_mut);
    ASSERT_MATRIX_NEAR(Z, X_mut, 1e-15);
}

void test_minmax_not_fitted() {
    Skigen::MinMaxScaler scaler;
    Eigen::MatrixXd X(2, 2);
    X << 1, 2, 3, 4;
    ASSERT_THROW(scaler.transform(X), std::runtime_error);
}

// ===================================================================
// MaxAbsScaler Tests
// ===================================================================

void test_maxabs_basic() {
    Eigen::MatrixXd X(4, 2);
    X << -3,  2,
          1, -4,
          2,  3,
         -1,  1;

    Skigen::MaxAbsScaler scaler;
    scaler.fit(X);
    Eigen::MatrixXd Z = scaler.transform(X);

    // All values in [-1, 1]
    ASSERT_TRUE(Z.cwiseAbs().maxCoeff() <= 1.0 + 1e-12);
    // max_abs should be [3, 4]
    ASSERT_NEAR(scaler.max_abs()(0), 3.0, 1e-12);
    ASSERT_NEAR(scaler.max_abs()(1), 4.0, 1e-12);
}

void test_maxabs_round_trip() {
    Eigen::MatrixXd X(3, 2);
    X << -5, 10, 3, -7, 0, 4;

    Skigen::MaxAbsScaler scaler;
    scaler.fit(X);
    Eigen::MatrixXd Z = scaler.transform(X);
    Eigen::MatrixXd X_back = scaler.inverse_transform(Z);
    ASSERT_MATRIX_NEAR(X, X_back, 1e-12);
}

void test_maxabs_zero_feature() {
    Eigen::MatrixXd X(3, 2);
    X << 0, 1, 0, 2, 0, 3;

    Skigen::MaxAbsScaler scaler;
    scaler.fit(X);
    // Zero-column: scale should be 1.0
    ASSERT_NEAR(scaler.scale()(0), 1.0, 1e-12);
}

// ===================================================================
// RobustScaler Tests
// ===================================================================

void test_robust_basic() {
    // X = [1, 2, 3, 4, 5] — median=3, IQR=Q75-Q25=4-2=2
    Eigen::MatrixXd X(5, 1);
    X << 1, 2, 3, 4, 5;

    Skigen::RobustScaler scaler;
    scaler.fit(X);

    ASSERT_NEAR(scaler.center()(0), 3.0, 1e-12);
    ASSERT_NEAR(scaler.scale()(0), 2.0, 1e-12);

    Eigen::MatrixXd Z = scaler.transform(X);
    // (3 - 3) / 2 = 0
    ASSERT_NEAR(Z(2, 0), 0.0, 1e-12);
}

void test_robust_round_trip() {
    Eigen::MatrixXd X(5, 2);
    X << 1, 10,
         2, 20,
         3, 30,
         4, 40,
         5, 50;

    Skigen::RobustScaler scaler;
    scaler.fit(X);
    Eigen::MatrixXd Z = scaler.transform(X);
    Eigen::MatrixXd X_back = scaler.inverse_transform(Z);
    ASSERT_MATRIX_NEAR(X, X_back, 1e-12);
}

void test_robust_no_centering() {
    Eigen::MatrixXd X(5, 1);
    X << 1, 2, 3, 4, 5;

    Skigen::RobustScaler<double> scaler(false, true);
    scaler.fit(X);
    Eigen::MatrixXd Z = scaler.transform(X);
    // No centering: Z(2,0) = 3 / 2 = 1.5
    ASSERT_NEAR(Z(2, 0), 3.0 / 2.0, 1e-12);
}

void test_robust_no_scaling() {
    Eigen::MatrixXd X(5, 1);
    X << 1, 2, 3, 4, 5;

    Skigen::RobustScaler<double> scaler(true, false);
    scaler.fit(X);
    Eigen::MatrixXd Z = scaler.transform(X);
    // Only centering: Z(0,0) = 1 - 3 = -2
    ASSERT_NEAR(Z(0, 0), -2.0, 1e-12);
}

// ===================================================================
// Normalizer Tests
// ===================================================================

void test_normalizer_l2() {
    Eigen::MatrixXd X(2, 3);
    X << 3, 4, 0,
         0, 0, 5;

    Skigen::Normalizer scaler(Skigen::Norm::L2);
    scaler.fit(X);
    Eigen::MatrixXd Z = scaler.transform(X);

    // Row 0: norm = 5, so [0.6, 0.8, 0]
    ASSERT_NEAR(Z(0, 0), 3.0 / 5.0, 1e-12);
    ASSERT_NEAR(Z(0, 1), 4.0 / 5.0, 1e-12);
    ASSERT_NEAR(Z(0, 2), 0.0, 1e-12);
    // Row 1: norm = 5, so [0, 0, 1]
    ASSERT_NEAR(Z(1, 2), 1.0, 1e-12);
}

void test_normalizer_l1() {
    Eigen::MatrixXd X(1, 3);
    X << 1, 2, 3;

    Skigen::Normalizer scaler(Skigen::Norm::L1);
    scaler.fit(X);
    Eigen::MatrixXd Z = scaler.transform(X);

    // sum = 6
    ASSERT_NEAR(Z(0, 0), 1.0 / 6.0, 1e-12);
    ASSERT_NEAR(Z(0, 1), 2.0 / 6.0, 1e-12);
    ASSERT_NEAR(Z(0, 2), 3.0 / 6.0, 1e-12);
}

void test_normalizer_max() {
    Eigen::MatrixXd X(1, 3);
    X << -5, 2, 3;

    Skigen::Normalizer scaler(Skigen::Norm::Max);
    scaler.fit(X);
    Eigen::MatrixXd Z = scaler.transform(X);

    ASSERT_NEAR(Z(0, 0), -1.0, 1e-12);
    ASSERT_NEAR(Z(0, 1), 2.0 / 5.0, 1e-12);
}

void test_normalizer_zero_row() {
    Eigen::MatrixXd X(2, 2);
    X << 0, 0,
         1, 0;

    Skigen::Normalizer scaler(Skigen::Norm::L2);
    scaler.fit(X);
    Eigen::MatrixXd Z = scaler.transform(X);
    // Zero row stays zero
    ASSERT_NEAR(Z(0, 0), 0.0, 1e-12);
    ASSERT_NEAR(Z(0, 1), 0.0, 1e-12);
}

void test_normalizer_inverse_throws() {
    Skigen::Normalizer scaler;
    Eigen::MatrixXd X(1, 2);
    X << 1, 2;
    scaler.fit(X);
    ASSERT_THROW(scaler.inverse_transform(X), std::runtime_error);
}

void test_normalizer_sparse_matches_dense_l2() {
    Eigen::MatrixXd Xd(4, 3);
    Xd << 3, 4, 0,
          0, 1, 0,
          0, 0, 5,
          1, 2, 2;
    Eigen::SparseMatrix<double> Xs = Xd.sparseView();

    Skigen::Normalizer<double> nz_d(Skigen::Norm::L2);
    Skigen::Normalizer<double> nz_s(Skigen::Norm::L2);
    nz_d.fit(Xd);
    nz_s.fit(Xs);

    Eigen::MatrixXd Yd = nz_d.transform(Xd);
    Eigen::SparseMatrix<double> Ys = nz_s.transform(Xs);
    Eigen::MatrixXd Ys_dense = Eigen::MatrixXd(Ys);

    ASSERT_TRUE(Yd.rows() == Ys_dense.rows());
    ASSERT_TRUE(Yd.cols() == Ys_dense.cols());
    for (int i = 0; i < Yd.rows(); ++i)
        for (int j = 0; j < Yd.cols(); ++j)
            ASSERT_NEAR(Yd(i, j), Ys_dense(i, j), 1e-12);
}

void test_normalizer_sparse_l1_and_max() {
    Eigen::MatrixXd Xd(2, 4);
    Xd << 1, -2, 3, 0,
          0,  4, 0, -1;
    Eigen::SparseMatrix<double> Xs = Xd.sparseView();

    // L1
    Skigen::Normalizer<double> nz_d_l1(Skigen::Norm::L1);
    Skigen::Normalizer<double> nz_s_l1(Skigen::Norm::L1);
    nz_d_l1.fit(Xd); nz_s_l1.fit(Xs);
    Eigen::MatrixXd Yd_l1 = nz_d_l1.transform(Xd);
    Eigen::MatrixXd Ys_l1 = Eigen::MatrixXd(nz_s_l1.transform(Xs));
    for (int i = 0; i < Yd_l1.rows(); ++i)
        for (int j = 0; j < Yd_l1.cols(); ++j)
            ASSERT_NEAR(Yd_l1(i, j), Ys_l1(i, j), 1e-12);

    // Max
    Skigen::Normalizer<double> nz_d_m(Skigen::Norm::Max);
    Skigen::Normalizer<double> nz_s_m(Skigen::Norm::Max);
    nz_d_m.fit(Xd); nz_s_m.fit(Xs);
    Eigen::MatrixXd Yd_m = nz_d_m.transform(Xd);
    Eigen::MatrixXd Ys_m = Eigen::MatrixXd(nz_s_m.transform(Xs));
    for (int i = 0; i < Yd_m.rows(); ++i)
        for (int j = 0; j < Yd_m.cols(); ++j)
            ASSERT_NEAR(Yd_m(i, j), Ys_m(i, j), 1e-12);
}

void test_normalizer_sparse_zero_row_left_alone() {
    Eigen::MatrixXd Xd(2, 3);
    Xd << 0, 0, 0,
          3, 4, 0;
    Eigen::SparseMatrix<double> Xs = Xd.sparseView();

    Skigen::Normalizer<double> nz(Skigen::Norm::L2);
    nz.fit(Xs);
    Eigen::SparseMatrix<double> Ys = nz.transform(Xs);
    Eigen::MatrixXd Yd = Eigen::MatrixXd(Ys);
    // Row 0 stays all zeros; row 1 becomes (0.6, 0.8, 0).
    ASSERT_NEAR(Yd(0, 0), 0.0, 1e-12);
    ASSERT_NEAR(Yd(0, 1), 0.0, 1e-12);
    ASSERT_NEAR(Yd(1, 0), 0.6, 1e-12);
    ASSERT_NEAR(Yd(1, 1), 0.8, 1e-12);
}

// ===================================================================
// LabelEncoder Tests
// ===================================================================

void test_label_encoder_basic() {
    Eigen::VectorXi y(5);
    y << 3, 1, 2, 1, 3;

    Skigen::LabelEncoder<int> enc;
    enc.fit(y);

    ASSERT_TRUE(enc.n_classes() == 3);
    // classes should be [1, 2, 3]
    ASSERT_TRUE(enc.classes()[0] == 1);
    ASSERT_TRUE(enc.classes()[1] == 2);
    ASSERT_TRUE(enc.classes()[2] == 3);

    Eigen::VectorXi encoded = enc.transform(y);
    // 3->2, 1->0, 2->1, 1->0, 3->2
    ASSERT_TRUE(encoded(0) == 2);
    ASSERT_TRUE(encoded(1) == 0);
    ASSERT_TRUE(encoded(2) == 1);
}

void test_label_encoder_round_trip() {
    Eigen::VectorXi y(4);
    y << 10, 20, 30, 20;

    Skigen::LabelEncoder<int> enc;
    Eigen::VectorXi encoded = enc.fit_transform(y);
    Eigen::VectorXi decoded = enc.inverse_transform(encoded);

    for (Eigen::Index i = 0; i < y.size(); ++i) {
        ASSERT_TRUE(decoded(i) == y(i));
    }
}

void test_label_encoder_unseen_label() {
    Eigen::VectorXi y(3);
    y << 1, 2, 3;

    Skigen::LabelEncoder<int> enc;
    enc.fit(y);

    Eigen::VectorXi y_bad(1);
    y_bad << 99;
    ASSERT_THROW(enc.transform(y_bad), std::invalid_argument);
}

void test_label_encoder_not_fitted() {
    Skigen::LabelEncoder<int> enc;
    Eigen::VectorXi y(1);
    y << 1;
    ASSERT_THROW(enc.transform(y), std::runtime_error);
}

// ===================================================================
// partial_fit (streaming) for MinMaxScaler and MaxAbsScaler
// ===================================================================

void test_minmax_partial_fit_first_call_equals_fit() {
    Eigen::MatrixXd X(5, 2);
    X << 1, 10, 2, 20, 3, 30, 4, 40, 5, 50;

    Skigen::MinMaxScaler<double> a, b;
    a.fit(X);
    b.partial_fit(X);

    for (int j = 0; j < 2; ++j) {
        ASSERT_NEAR(a.data_min()(j), b.data_min()(j), 1e-12);
        ASSERT_NEAR(a.data_max()(j), b.data_max()(j), 1e-12);
        ASSERT_NEAR(a.scale()(j),    b.scale()(j),    1e-12);
        ASSERT_NEAR(a.min()(j),      b.min()(j),      1e-12);
    }
}

void test_minmax_partial_fit_batched_equals_monolithic() {
    Eigen::MatrixXd X(8, 2);
    X << 1, 10, -2, 25, 3, 7, 4, 40, 5, 50, -6, 12, 7, 70, 0, 5;

    Skigen::MinMaxScaler<double> mono;
    mono.fit(X);

    Skigen::MinMaxScaler<double> stream;
    stream.partial_fit(X.topRows(3));
    stream.partial_fit(X.middleRows(3, 3));
    stream.partial_fit(X.bottomRows(2));

    for (int j = 0; j < 2; ++j) {
        ASSERT_NEAR(mono.data_min()(j), stream.data_min()(j), 1e-12);
        ASSERT_NEAR(mono.data_max()(j), stream.data_max()(j), 1e-12);
        ASSERT_NEAR(mono.scale()(j),    stream.scale()(j),    1e-12);
    }
    Eigen::MatrixXd Zm = mono.transform(X);
    Eigen::MatrixXd Zs = stream.transform(X);
    for (int i = 0; i < X.rows(); ++i) for (int j = 0; j < 2; ++j) {
        ASSERT_NEAR(Zm(i, j), Zs(i, j), 1e-12);
    }
}

void test_minmax_partial_fit_feature_mismatch_throws() {
    Eigen::MatrixXd X1(3, 2); X1 << 1, 2, 3, 4, 5, 6;
    Eigen::MatrixXd X2(2, 3); X2.setOnes();
    Skigen::MinMaxScaler<double> sc;
    sc.partial_fit(X1);
    ASSERT_THROW(sc.partial_fit(X2), std::invalid_argument);
}

void test_maxabs_partial_fit_batched_equals_monolithic() {
    Eigen::MatrixXd X(6, 3);
    X << 1, -5, 2,
         3,  4, -6,
        -7,  2, 1,
         8, -1, 0,
        -2,  9, 4,
         5,  3, -3;

    Skigen::MaxAbsScaler<double> mono;
    mono.fit(X);

    Skigen::MaxAbsScaler<double> stream;
    stream.partial_fit(X.topRows(2));
    stream.partial_fit(X.middleRows(2, 2));
    stream.partial_fit(X.bottomRows(2));

    for (int j = 0; j < 3; ++j) {
        ASSERT_NEAR(mono.max_abs()(j), stream.max_abs()(j), 1e-12);
        ASSERT_NEAR(mono.scale()(j),   stream.scale()(j),   1e-12);
    }
    Eigen::MatrixXd Zm = mono.transform(X);
    Eigen::MatrixXd Zs = stream.transform(X);
    for (int i = 0; i < X.rows(); ++i) for (int j = 0; j < 3; ++j) {
        ASSERT_NEAR(Zm(i, j), Zs(i, j), 1e-12);
    }
}

void test_maxabs_partial_fit_feature_mismatch_throws() {
    Eigen::MatrixXd X1(2, 2); X1 << 1, 2, 3, 4;
    Eigen::MatrixXd X2(2, 4); X2.setOnes();
    Skigen::MaxAbsScaler<double> sc;
    sc.partial_fit(X1);
    ASSERT_THROW(sc.partial_fit(X2), std::invalid_argument);
}

// ===================================================================
// Main
// ===================================================================

int main() {
    std::cout << "=== MinMaxScaler Tests ===\n";
    run_test("minmax_basic",             test_minmax_basic);
    run_test("minmax_round_trip",        test_minmax_round_trip);
    run_test("minmax_custom_range",      test_minmax_custom_range);
    run_test("minmax_constant_feature",  test_minmax_constant_feature);
    run_test("minmax_inplace",           test_minmax_inplace);
    run_test("minmax_not_fitted",        test_minmax_not_fitted);

    std::cout << "\n=== MaxAbsScaler Tests ===\n";
    run_test("maxabs_basic",             test_maxabs_basic);
    run_test("maxabs_round_trip",        test_maxabs_round_trip);
    run_test("maxabs_zero_feature",      test_maxabs_zero_feature);

    std::cout << "\n=== RobustScaler Tests ===\n";
    run_test("robust_basic",             test_robust_basic);
    run_test("robust_round_trip",        test_robust_round_trip);
    run_test("robust_no_centering",      test_robust_no_centering);
    run_test("robust_no_scaling",        test_robust_no_scaling);

    std::cout << "\n=== Normalizer Tests ===\n";
    run_test("normalizer_l2",            test_normalizer_l2);
    run_test("normalizer_l1",            test_normalizer_l1);
    run_test("normalizer_max",           test_normalizer_max);
    run_test("normalizer_zero_row",      test_normalizer_zero_row);
    run_test("normalizer_inverse_throws", test_normalizer_inverse_throws);
    run_test("normalizer_sparse_matches_dense_l2",   test_normalizer_sparse_matches_dense_l2);
    run_test("normalizer_sparse_l1_and_max",         test_normalizer_sparse_l1_and_max);
    run_test("normalizer_sparse_zero_row_left_alone", test_normalizer_sparse_zero_row_left_alone);

    std::cout << "\n=== LabelEncoder Tests ===\n";
    run_test("label_encoder_basic",        test_label_encoder_basic);
    run_test("label_encoder_round_trip",   test_label_encoder_round_trip);
    run_test("label_encoder_unseen_label", test_label_encoder_unseen_label);
    run_test("label_encoder_not_fitted",   test_label_encoder_not_fitted);

    std::cout << "\n=== partial_fit (streaming) Tests ===\n";
    run_test("minmax_partial_fit_first_call_equals_fit",     test_minmax_partial_fit_first_call_equals_fit);
    run_test("minmax_partial_fit_batched_equals_monolithic", test_minmax_partial_fit_batched_equals_monolithic);
    run_test("minmax_partial_fit_feature_mismatch_throws",   test_minmax_partial_fit_feature_mismatch_throws);
    run_test("maxabs_partial_fit_batched_equals_monolithic", test_maxabs_partial_fit_batched_equals_monolithic);
    run_test("maxabs_partial_fit_feature_mismatch_throws",   test_maxabs_partial_fit_feature_mismatch_throws);

    std::cout << "\n" << g_passed << " passed, " << g_failed << " failed.\n";
    return g_failed > 0 ? 1 : 0;
}
