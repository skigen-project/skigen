// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_SVM_NU_SVC_H
#define SKIGEN_SVM_NU_SVC_H

#include "../Core/Base.h"
#include "Detail/Kernels.h"

#include <stdexcept>
#include <string>

namespace Skigen {

/// @addtogroup Algo_KernelSVM
/// @{

/// @brief Nu-SVM classifier — placeholder.
///
/// Mirrors the constructor signature of
/// [sklearn.svm.NuSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html)
/// so user code can be written against the API surface, but `fit()`
/// throws: the nu-formulation requires libsvm's specialised SMO
/// variant, which is not implemented. Use Skigen::SVC (C-SVM) instead.
template <typename Scalar = double>
class NuSVC : public Classifier<NuSVC<Scalar>, Scalar> {
public:
    using Base = Classifier<NuSVC<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::IndexType;
    using typename Base::ScalarType;
    using typename Base::LabelType;
    using Kernel = internal::KernelKind;

    explicit NuSVC(Scalar nu = Scalar{0.5},
                   Kernel kernel = Kernel::RBF,
                   int degree = 3,
                   Scalar gamma = Scalar{0},
                   Scalar coef0 = Scalar{0})
        : nu_(nu), kernel_(kernel), degree_(degree),
          gamma_(gamma), coef0_(coef0) {}

    [[nodiscard]] Scalar nu() const noexcept { return nu_; }

    SKIGEN_PARAMS(
        (nu,     nu_,     double),
        (degree, degree_, int),
        (gamma,  gamma_,  double),
        (coef0,  coef0_,  double))

    NuSVC& fit_impl(const Eigen::Ref<const MatrixType>& /*X*/,
                    const Eigen::Ref<const Eigen::VectorXi>& /*y*/) {
        throw std::runtime_error(
            "NuSVC: the nu-SVM solver is not implemented. "
            "Use Skigen::SVC (C-SVM) instead.");
    }

    [[nodiscard]] LabelType predict_impl(
        const Eigen::Ref<const MatrixType>& /*X*/) const {
        throw std::runtime_error("NuSVC: not implemented.");
    }

private:
    Scalar nu_;
    Kernel kernel_;
    int degree_;
    Scalar gamma_;
    Scalar coef0_;
};

/// @}

}  // namespace Skigen

#endif  // SKIGEN_SVM_NU_SVC_H
