// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors
//
// Parity for the manifold learners via trustworthiness — a rotation- and
// reflection-invariant measure of local-neighbourhood preservation. Skigen's
// embedding must reach trustworthiness within a band of scikit-learn's
// (or, for UMAP, above a fixed floor, since umap-learn is not required).

#include <Skigen/Dense>

#include <algorithm>
#include <numeric>
#include <optional>
#include <vector>

#include "parity_common.h"

namespace {
using namespace skigen_parity;

// Rank-based trustworthiness (Venna & Kaski), matching sklearn's definition.
double trustworthiness(const Eigen::MatrixXd& X, const Eigen::MatrixXd& emb,
                       int k) {
    const Eigen::Index n = X.rows();
    auto pairwise = [](const Eigen::MatrixXd& M) {
        Eigen::MatrixXd D(M.rows(), M.rows());
        for (Eigen::Index i = 0; i < M.rows(); ++i)
            for (Eigen::Index j = 0; j < M.rows(); ++j)
                D(i, j) = (M.row(i) - M.row(j)).norm();
        return D;
    };
    Eigen::MatrixXd Dx = pairwise(X);
    Eigen::MatrixXd De = pairwise(emb);

    // rank_in_X(i, j): rank of j among i's neighbours in the original space
    // (1 = nearest non-self).
    std::vector<std::vector<int>> rank_in_X(static_cast<std::size_t>(n));
    for (Eigen::Index i = 0; i < n; ++i) {
        std::vector<int> order(static_cast<std::size_t>(n));
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](int a, int b) { return Dx(i, a) < Dx(i, b); });
        rank_in_X[static_cast<std::size_t>(i)].assign(
            static_cast<std::size_t>(n), 0);
        int r = 0;
        for (int idx : order) {
            if (idx == static_cast<int>(i)) continue;
            rank_in_X[static_cast<std::size_t>(i)][static_cast<std::size_t>(idx)]
                = ++r;
        }
    }

    double sum = 0.0;
    for (Eigen::Index i = 0; i < n; ++i) {
        std::vector<int> order(static_cast<std::size_t>(n));
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](int a, int b) { return De(i, a) < De(i, b); });
        int taken = 0;
        for (int idx : order) {
            if (idx == static_cast<int>(i)) continue;
            ++taken;
            const int rank =
                rank_in_X[static_cast<std::size_t>(i)]
                         [static_cast<std::size_t>(idx)];
            if (rank > k) sum += rank - k;
            if (taken == k) break;
        }
    }
    const double nn = static_cast<double>(n);
    const double kk = static_cast<double>(k);
    return 1.0 - (2.0 / (nn * kk * (2.0 * nn - 3.0 * kk - 1.0))) * sum;
}

template <typename Embedder>
void check(const std::string& e, Embedder embedder, double band) {
    Eigen::MatrixXd X = load_matrix(e, "X.csv");
    const double ref = load_vector(e, "trust.csv")(0);
    Eigen::MatrixXd emb = embedder.fit_transform(X);
    if (emb.rows() != X.rows() || emb.cols() != 2)
        throw ParityFailure(e + ": unexpected embedding shape");
    const double got = trustworthiness(X, emb, 5);
    expect_score(got, ref, band, e + ".trustworthiness");
}
}  // namespace

void parity_manifold() {
    run("Isomap", [] {
        check("isomap", Skigen::Isomap<double>(2, 5), 0.10);
    });
    run("MDS", [] {
        check("mds", Skigen::MDS<double>(2, 300, 1e-4, true, 0), 0.12);
    });
    run("LocallyLinearEmbedding", [] {
        check("locally_linear_embedding",
              Skigen::LocallyLinearEmbedding<double>(2, 5), 0.12);
    });
    run("SpectralEmbedding", [] {
        check("spectral_embedding", Skigen::SpectralEmbedding<double>(2, 5),
              0.12);
    });
    run("TSNE", [] {
        check("tsne",
              Skigen::TSNE<double>(2, 15.0, 200.0, 300, "exact", 0.5, 12.0,
                                   std::optional<uint64_t>(0)),
              0.15);
    });
    run("UMAP", [] {
        check("umap", Skigen::UMAP<double>(2, 15, 0.1, 1.0, 200, 5, 0), 0.20);
    });
}
