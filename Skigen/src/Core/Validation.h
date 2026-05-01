// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_CORE_VALIDATION_H
#define SKIGEN_CORE_VALIDATION_H

#include <Eigen/Core>
#include <stdexcept>
#include <string>

namespace Skigen {
namespace internal {

template <typename MatrixType>
void check_non_empty(const MatrixType& X, const std::string& name = "X") {
    if (X.rows() == 0 || X.cols() == 0) {
        throw std::invalid_argument(
            name + " must have at least one sample and one feature. "
            "Got shape (" + std::to_string(X.rows()) + ", " +
            std::to_string(X.cols()) + ").");
    }
}

template <typename MatrixType, typename VectorType>
void check_consistent_length(const MatrixType& X, const VectorType& y) {
    if (X.rows() != y.rows()) {
        throw std::invalid_argument(
            "Found input variables with inconsistent numbers of samples: "
            "X has " + std::to_string(X.rows()) + " samples, "
            "y has " + std::to_string(y.rows()) + " samples.");
    }
}

} // namespace internal
} // namespace Skigen

#endif // SKIGEN_CORE_VALIDATION_H
