// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#include <Skigen/Dense>

#include <cmath>
#include <functional>
#include <iostream>
#include <numbers>
#include <sstream>
#include <string>

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
// Helper: generate a small 2D dataset (three blobs)
// ===================================================================

static Eigen::MatrixXd make_blobs(int n = 30) {
    Eigen::MatrixXd X(n, 3);
    for (int i = 0; i < n; ++i) {
        double t = static_cast<double>(i) / n;
        X(i, 0) = std::cos(2.0 * std::numbers::pi * t) + 0.1 * (i % 3);
        X(i, 1) = std::sin(2.0 * std::numbers::pi * t) + 0.1 * ((i + 1) % 3);
        X(i, 2) = t * 2.0 - 1.0;
    }
    return X;
}

// ===================================================================
// MDS Tests
// ===================================================================

void test_mds_basic() {
    auto X = make_blobs(20);
    Skigen::MDS<double> mds(2, 50, 1e-3, true, 42);
    auto Y = mds.fit_transform(X);
    ASSERT_TRUE(Y.rows() == 20);
    ASSERT_TRUE(Y.cols() == 2);
    ASSERT_TRUE(mds.stress() >= 0.0);
    ASSERT_TRUE(mds.n_iter() > 0);
}

void test_mds_preserves_distances() {
    Eigen::MatrixXd X(4, 2);
    X << 0, 0, 1, 0, 0, 1, 1, 1;
    Skigen::MDS<double> mds(2, 300, 1e-6, true, 42);
    auto Y = mds.fit_transform(X);
    ASSERT_TRUE(Y.rows() == 4);
    ASSERT_TRUE(Y.cols() == 2);
    ASSERT_TRUE(mds.stress() < 0.5);
}

void test_mds_n_init_keeps_best_stress() {
    auto X = make_blobs(18);
    Skigen::MDS<double> single(2, 100, 1e-4, true, 7, 1);
    Skigen::MDS<double> multi(2, 100, 1e-4, true, 7, 4);
    auto single_embedding = single.fit_transform(X);
    auto multi_embedding = multi.fit_transform(X);
    ASSERT_TRUE(single_embedding.rows() == multi_embedding.rows());
    ASSERT_TRUE(single_embedding.cols() == multi_embedding.cols());
    ASSERT_TRUE(multi.n_init() == 4);
    ASSERT_TRUE(multi.n_iter() > 0);
    ASSERT_TRUE(multi.stress() <= single.stress() + 1e-12);
}

void test_mds_invalid_parameters_throw() {
    auto X = make_blobs(8);
    bool threw = false;
    try { Skigen::MDS<double>(2, 100, 1e-3, false, 42).fit(X); }
    catch (const std::invalid_argument&) { threw = true; }
    ASSERT_TRUE(threw);

    threw = false;
    try { Skigen::MDS<double>(2, 100, 1e-3, true, 42, 0).fit(X); }
    catch (const std::invalid_argument&) { threw = true; }
    ASSERT_TRUE(threw);
}

// ===================================================================
// Isomap Tests
// ===================================================================

void test_isomap_basic() {
    auto X = make_blobs(25);
    Skigen::Isomap<double> iso(2, 5);
    auto Y = iso.fit_transform(X);
    ASSERT_TRUE(Y.rows() == 25);
    ASSERT_TRUE(Y.cols() == 2);
}

void test_isomap_dist_matrix() {
    auto X = make_blobs(15);
    Skigen::Isomap<double> iso(2, 4);
    iso.fit(X);
    auto& D = iso.dist_matrix();
    ASSERT_TRUE(D.rows() == 15);
    ASSERT_TRUE(D.cols() == 15);
    for (int i = 0; i < 15; ++i) {
        ASSERT_NEAR(D(i, i), 0.0, 1e-10);
    }
}

// ===================================================================
// TSNE Tests
// ===================================================================

void test_tsne_basic() {
    auto X = make_blobs(20);
    Skigen::TSNE<double> tsne(2, 5.0, 200.0, 300, 42);
    auto Y = tsne.fit_transform(X);
    ASSERT_TRUE(Y.rows() == 20);
    ASSERT_TRUE(Y.cols() == 2);
    ASSERT_TRUE(tsne.kl_divergence() >= 0.0);
}

// ===================================================================
// LLE Tests
// ===================================================================

void test_lle_basic() {
    auto X = make_blobs(20);
    Skigen::LocallyLinearEmbedding<double> lle(2, 5);
    auto Y = lle.fit_transform(X);
    ASSERT_TRUE(Y.rows() == 20);
    ASSERT_TRUE(Y.cols() == 2);
    ASSERT_TRUE(lle.reconstruction_error() >= 0.0);
}

// ===================================================================
// SpectralEmbedding Tests
// ===================================================================

void test_spectral_embedding_basic() {
    auto X = make_blobs(20);
    Skigen::SpectralEmbedding<double> se(2, 5);
    auto Y = se.fit_transform(X);
    ASSERT_TRUE(Y.rows() == 20);
    ASSERT_TRUE(Y.cols() == 2);
    auto& A = se.affinity_matrix();
    ASSERT_TRUE(A.rows() == 20);
    ASSERT_TRUE(A.cols() == 20);
}

// ===================================================================
// UMAP Tests
// ===================================================================

void test_umap_basic() {
    auto X = make_blobs(30);
    Skigen::UMAP<double> umap(2, 5, 0.1, 1.0, 100, 5, 42);
    auto Y = umap.fit_transform(X);
    ASSERT_TRUE(Y.rows() == 30);
    ASSERT_TRUE(Y.cols() == 2);
    bool has_nan = false;
    for (int i = 0; i < Y.rows(); ++i) {
        for (int j = 0; j < Y.cols(); ++j) {
            if (std::isnan(Y(i, j))) has_nan = true;
        }
    }
    ASSERT_TRUE(!has_nan);
}

// ===================================================================
// SKIGEN_PARAMS Tests
// ===================================================================

void test_manifold_params() {
    Skigen::MDS<double> mds;
    auto p = mds.get_params();
    ASSERT_TRUE(p.size() >= 3);
    ASSERT_TRUE(std::get<int>(p.at("n_components")) == 2);
    ASSERT_TRUE(std::get<int>(p.at("max_iter")) == 300);
    ASSERT_TRUE(std::get<int>(p.at("n_init")) == 4);

    Skigen::UMAP<double> umap;
    auto pu = umap.get_params();
    ASSERT_TRUE(std::get<int>(pu.at("n_neighbors")) == 15);
    ASSERT_NEAR(std::get<double>(pu.at("min_dist")), 0.1, 1e-10);
}

// ===================================================================

int main() {
    std::cout << "=== MDS Tests ===\n";
    run_test("mds_basic", test_mds_basic);
    run_test("mds_preserves_distances", test_mds_preserves_distances);
    run_test("mds_n_init_keeps_best_stress", test_mds_n_init_keeps_best_stress);
    run_test("mds_invalid_parameters_throw", test_mds_invalid_parameters_throw);

    std::cout << "\n=== Isomap Tests ===\n";
    run_test("isomap_basic", test_isomap_basic);
    run_test("isomap_dist_matrix", test_isomap_dist_matrix);

    std::cout << "\n=== TSNE Tests ===\n";
    run_test("tsne_basic", test_tsne_basic);

    std::cout << "\n=== LLE Tests ===\n";
    run_test("lle_basic", test_lle_basic);

    std::cout << "\n=== SpectralEmbedding Tests ===\n";
    run_test("spectral_embedding_basic", test_spectral_embedding_basic);

    std::cout << "\n=== UMAP Tests ===\n";
    run_test("umap_basic", test_umap_basic);

    std::cout << "\n=== SKIGEN_PARAMS Tests ===\n";
    run_test("manifold_params", test_manifold_params);

    std::cout << "\n" << g_passed << " passed, " << g_failed << " failed.\n";
    return g_failed > 0 ? 1 : 0;
}
