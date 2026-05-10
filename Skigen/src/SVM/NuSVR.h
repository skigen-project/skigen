// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_SVM_NU_SVR_H
#define SKIGEN_SVM_NU_SVR_H

#include "../Core/Base.h"
#include "Detail/Kernels.h"

#include <stdexcept>

namespace Skigen {

/// @addtogroup Algo_KernelSVM
/// @{

/// @brief Nu-SVR — placeholder. Use Skigen::SVR instead.
///
/// Mirrors
/// [sklearn.svm.NuSVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVR.html)
/// constructor signature so user code can be written against the API
/// surface, but `fit()` throws: the nu-formulation requires libsvm's
/// specialised SMO variant, which is not implemented.
template <typename Scalar = double>
class NuSVR : public Predictor<NuSVR<Scalar>, Scalar> {
public:
    using Base = Predictor<NuSVR<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::IndexType;
    using typename Base::ScalarType;
    using Kernel = internal::KernelKind;

    explicit NuSVR(Scalar nu = Scalar{0.5},
                   Scalar C  = Scalar{1.0},
                   Kernel kernel = Kernel::RBF,
                   int degree = 3,
                   Scalar gamma = Scalar{0},
                   Scalar coef0 = Scalar{0})
        : nu_(nu), C_(C), kernel_(kernel), degree_(degree),
          gamma_(gamma), coef0_(coef0) {}

    NuSVR& fit_impl(const Eigen::Ref<const MatrixType>& /*X*/,
                    const Eigen::Ref<const VectorType>& /*y*/) {
        throw std::runtime_error(
            "NuSVR: the nu-SVR solver is not implemented. "
            "Use Skigen::SVR (epsilon-SVR) instead.");
    }
    [[nodiscard]] VectorType predict_impl(
        const Eigen::Ref<const MatrixType>& /*X*/) const {
        throw std::runtime_error("NuSVR: not implemented.");
    }

private:
    Scalar nu_;
    Scalar C_;
    Kernel kernel_;
    int degree_;
    Scalar gamma_;
    Scalar coef0_;
};

/// @}

}  // namespace Skigen

#endif  // SKIGEN_SVM_NU_SVR_H
