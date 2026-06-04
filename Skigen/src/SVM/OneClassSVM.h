// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_SVM_ONE_CLASS_SVM_H
#define SKIGEN_SVM_ONE_CLASS_SVM_H

#include "../Core/Base.h"
#include "Detail/Kernels.h"

#include <stdexcept>

namespace Skigen {

/// @addtogroup Algo_KernelSVM
/// @{

/// @brief One-class SVM — placeholder.
///
/// Mirrors
/// [sklearn.svm.OneClassSVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html)
/// constructor signature so user code can be written against the API
/// surface, but `fit()` throws: the one-class formulation requires the
/// same nu-SVM SMO variant that backs NuSVC/NuSVR, which is not
/// implemented.
template <typename Scalar = double>
class OneClassSVM : public Estimator<OneClassSVM<Scalar>, Scalar> {
public:
    using Base = Estimator<OneClassSVM<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::IndexType;
    using typename Base::ScalarType;
    using Kernel = internal::KernelKind;

    explicit OneClassSVM(Kernel kernel = Kernel::RBF,
                         int degree = 3,
                         Scalar gamma = Scalar{0},
                         Scalar coef0 = Scalar{0},
                         Scalar nu = Scalar{0.5})
        : kernel_(kernel), degree_(degree),
          gamma_(gamma), coef0_(coef0), nu_(nu) {}

    SKIGEN_PARAMS(
        (degree, degree_, int),
        (gamma,  gamma_,  double),
        (coef0,  coef0_,  double),
        (nu,     nu_,     double))

    OneClassSVM& fit(const Eigen::Ref<const MatrixType>& /*X*/) {
        throw std::runtime_error(
            "OneClassSVM: not implemented (requires the nu-SVM SMO "
            "variant that NuSVC/NuSVR also depend on).");
    }
    [[nodiscard]] Eigen::VectorXi predict(
        const Eigen::Ref<const MatrixType>& /*X*/) const {
        throw std::runtime_error("OneClassSVM: not implemented.");
    }

private:
    Kernel kernel_;
    int degree_;
    Scalar gamma_;
    Scalar coef0_;
    Scalar nu_;
};

/// @}

}  // namespace Skigen

#endif  // SKIGEN_SVM_ONE_CLASS_SVM_H
