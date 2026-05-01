// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_CORE_TRAITS_H
#define SKIGEN_CORE_TRAITS_H

#include <Eigen/Core>
#include <type_traits>

namespace Skigen {

// ---------------------------------------------------------------------------
// Canonical type aliases for all Skigen estimators
// ---------------------------------------------------------------------------

template <typename Scalar>
struct EigenTypes {
    static_assert(std::is_floating_point_v<Scalar>,
                  "Scalar must be a floating-point type");

    using ScalarType    = Scalar;
    using MatrixType    = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorType    = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using RowVectorType = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;
    using IndexType     = Eigen::Index;
};

} // namespace Skigen

#endif // SKIGEN_CORE_TRAITS_H
