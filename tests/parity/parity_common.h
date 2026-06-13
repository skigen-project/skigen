// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors
//
// Shared harness for the Skigen parity tests. Each parity check loads the
// CSV fixtures produced by tests/parity/generate_<estimator>_reference.py and
// compares Skigen's fitted attributes / predictions against the recorded
// scikit-learn reference within a per-estimator tolerance.

#ifndef SKIGEN_PARITY_COMMON_H
#define SKIGEN_PARITY_COMMON_H

#include <Eigen/Dense>

#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef SKIGEN_PARITY_DATA_DIR
#error "SKIGEN_PARITY_DATA_DIR must be defined (path to tests/parity/data)."
#endif

namespace skigen_parity {

inline int g_passed = 0;
inline int g_failed = 0;

struct ParityFailure : std::exception {
    std::string msg;
    explicit ParityFailure(std::string m) : msg(std::move(m)) {}
    const char* what() const noexcept override { return msg.c_str(); }
};

// --- CSV loading ----------------------------------------------------------

inline std::string data_path(const std::string& estimator,
                             const std::string& file) {
    return std::string(SKIGEN_PARITY_DATA_DIR) + "/" + estimator + "/" + file;
}

inline Eigen::MatrixXd load_matrix(const std::string& estimator,
                                   const std::string& file) {
    std::ifstream in(data_path(estimator, file));
    if (!in) {
        throw ParityFailure("cannot open fixture: " +
                            data_path(estimator, file));
    }
    std::vector<std::vector<double>> rows;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            row.push_back(std::stod(cell));
        }
        rows.push_back(std::move(row));
    }
    if (rows.empty()) {
        throw ParityFailure("empty fixture: " + data_path(estimator, file));
    }
    const Eigen::Index r = static_cast<Eigen::Index>(rows.size());
    const Eigen::Index c = static_cast<Eigen::Index>(rows[0].size());
    Eigen::MatrixXd M(r, c);
    for (Eigen::Index i = 0; i < r; ++i) {
        for (Eigen::Index j = 0; j < c; ++j) {
            M(i, j) = rows[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
        }
    }
    return M;
}

inline Eigen::VectorXd load_vector(const std::string& estimator,
                                   const std::string& file) {
    Eigen::MatrixXd M = load_matrix(estimator, file);
    if (M.rows() == 1) return M.row(0).transpose();
    if (M.cols() == 1) return M.col(0);
    throw ParityFailure("expected a vector fixture: " +
                        data_path(estimator, file));
}

// --- assertions -----------------------------------------------------------

inline void expect_near(double a, double b, double tol, const std::string& ctx) {
    if (std::isnan(a) || std::isnan(b) || std::abs(a - b) > tol) {
        std::ostringstream oss;
        oss << ctx << ": " << a << " vs " << b << " (tol=" << tol
            << ", diff=" << std::abs(a - b) << ")";
        throw ParityFailure(oss.str());
    }
}

template <typename A, typename B>
void expect_allclose(const A& got, const B& ref, double tol,
                     const std::string& ctx) {
    if (got.rows() != ref.rows() || got.cols() != ref.cols()) {
        std::ostringstream oss;
        oss << ctx << ": shape (" << got.rows() << "," << got.cols()
            << ") vs (" << ref.rows() << "," << ref.cols() << ")";
        throw ParityFailure(oss.str());
    }
    for (Eigen::Index i = 0; i < got.rows(); ++i) {
        for (Eigen::Index j = 0; j < got.cols(); ++j) {
            expect_near(got(i, j), ref(i, j), tol, ctx);
        }
    }
}

// Compare two embeddings/components up to per-column sign flips (eigvec /
// SVD sign ambiguity). Both must already share the same shape.
template <typename A, typename B>
void expect_allclose_signed(const A& got, const B& ref, double tol,
                            const std::string& ctx) {
    if (got.rows() != ref.rows() || got.cols() != ref.cols()) {
        std::ostringstream oss;
        oss << ctx << ": shape (" << got.rows() << "," << got.cols()
            << ") vs (" << ref.rows() << "," << ref.cols() << ")";
        throw ParityFailure(oss.str());
    }
    for (Eigen::Index j = 0; j < got.cols(); ++j) {
        const double dpos = (got.col(j) - ref.col(j)).cwiseAbs().maxCoeff();
        const double dneg = (got.col(j) + ref.col(j)).cwiseAbs().maxCoeff();
        const double best = std::min(dpos, dneg);
        if (best > tol) {
            std::ostringstream oss;
            oss << ctx << " (col " << j << "): max|diff|=" << best
                << " > tol=" << tol;
            throw ParityFailure(oss.str());
        }
    }
}

// --- behavioural-parity metrics ------------------------------------------
//
// Estimators with documented algorithmic differences from scikit-learn
// (tree ensembles, SMO-based SVMs, mini-batch MLPs, manifold learners) cannot
// match fitted attributes bit-for-bit. For those, parity is asserted on the
// task metric: Skigen must reach a held-out score within a band of sklearn's.

inline double r2_score(const Eigen::VectorXd& pred, const Eigen::VectorXd& y) {
    const double mean = y.mean();
    const double ss_res = (y - pred).squaredNorm();
    const double ss_tot = (y.array() - mean).matrix().squaredNorm();
    return ss_tot > 0.0 ? 1.0 - ss_res / ss_tot : 0.0;
}

inline double accuracy(const Eigen::VectorXi& pred, const Eigen::VectorXi& y) {
    Eigen::Index correct = 0;
    for (Eigen::Index i = 0; i < y.size(); ++i)
        if (pred(i) == y(i)) ++correct;
    return static_cast<double>(correct) / static_cast<double>(y.size());
}

// Skigen's score must be at least sklearn's minus `band` (it may exceed it).
inline void expect_score(double got, double ref, double band,
                         const std::string& ctx) {
    if (got < ref - band) {
        std::ostringstream oss;
        oss << ctx << ": score " << got << " < sklearn " << ref << " - "
            << band;
        throw ParityFailure(oss.str());
    }
}

inline Eigen::VectorXi to_int(const Eigen::VectorXd& v) {
    Eigen::VectorXi out(v.size());
    for (Eigen::Index i = 0; i < v.size(); ++i)
        out(i) = static_cast<int>(std::lround(v(i)));
    return out;
}

inline void run(const std::string& name, const std::function<void()>& fn) {
    try {
        fn();
        ++g_passed;
        std::cout << "  PASS  " << name << "\n";
    } catch (const std::exception& e) {
        ++g_failed;
        std::cout << "  FAIL  " << name << "\n        " << e.what() << "\n";
    }
}

inline int summary() {
    std::cout << "\nParity: " << g_passed << " passed, " << g_failed
              << " failed\n";
    return g_failed == 0 ? 0 : 1;
}

}  // namespace skigen_parity

#endif  // SKIGEN_PARITY_COMMON_H
