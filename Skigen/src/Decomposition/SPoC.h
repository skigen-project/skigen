// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_DECOMPOSITION_SPOC_H
#define SKIGEN_DECOMPOSITION_SPOC_H

#include "../Core/Base.h"
#include "../Core/Validation.h"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace Skigen {

/// @defgroup Algo_SPoC SPoC
/// @ingroup Decomposition
/// @brief Source Power Comodulation (SPoC) spatial filter.
/// @{

/// @brief Source Power Comodulation (SPoC).
///
/// Finds spatial filters whose band-power envelope maximally covaries
/// with a continuous target variable. Used for regression-based BCI and
/// neural decoding of continuous states.
///
/// ### Algorithm
///
/// 1. Normalize the target @f$z@f$ to zero mean and unit variance.
/// 2. Compute per-epoch covariance @f$C_e@f$ and accumulate:
///    @f$C = \frac{1}{E}\sum C_e@f$, @f$C_z = \frac{1}{E}\sum z_e C_e@f$.
/// 3. Solve the generalized eigenvalue problem @f$C_z w = \lambda C w@f$.
/// 4. Select components by largest absolute eigenvalue.
///
/// Mirrors
/// [mne.decoding.SPoC](https://mne.tools/stable/generated/mne.decoding.SPoC.html).
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `n_components` | `int` | 6 | Number of components to extract. |
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type | Description |
/// |----------|------|-------------|
/// | `filters()` | `MatrixType` | Spatial filters (n_components × n_channels). |
/// | `patterns()` | `MatrixType` | Spatial patterns (n_components × n_channels). |
/// | `eigenvalues()` | `VectorType` | Eigenvalues for selected components. |
///
template <typename Scalar = double>
class SPoC
    : public Estimator<SPoC<Scalar>, Scalar> {
public:
    using Base = Estimator<SPoC<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;
    using typename Base::ScalarType;

    /// @brief Construct a SPoC estimator.
    explicit SPoC(int n_components = 6)
        : n_components_(n_components) {}

    // -- Accessors ----------------------------------------------------------

    /// @brief Spatial filters (n_components × n_channels).
    [[nodiscard]] const MatrixType& filters() const {
        this->check_is_fitted();
        return filters_;
    }

    /// @brief Spatial patterns (n_components × n_channels).
    [[nodiscard]] const MatrixType& patterns() const {
        this->check_is_fitted();
        return patterns_;
    }

    /// @brief Eigenvalues for the selected components.
    [[nodiscard]] const VectorType& eigenvalues() const {
        this->check_is_fitted();
        return eigenvalues_;
    }

    // -- fit ----------------------------------------------------------------

    /// @brief Fit SPoC from epoch data and a continuous target.
    ///
    /// Each epoch has shape (n_channels × n_times). The target vector has
    /// one entry per epoch.
    ///
    /// @param epochs  Vector of epoch matrices.
    /// @param target  Continuous target variable (one per epoch).
    /// @return Reference to the fitted estimator.
    SPoC& fit(const std::vector<MatrixType>& epochs,
              const Eigen::Ref<const VectorType>& target) {
        if (epochs.empty()) {
            throw std::invalid_argument("SPoC::fit: epochs must be non-empty");
        }
        const auto n_epochs = static_cast<Eigen::Index>(epochs.size());
        if (target.size() != n_epochs) {
            throw std::invalid_argument(
                "SPoC::fit: target size must match epoch count");
        }

        const auto n_ch = epochs[0].rows();
        this->n_features_in_ = n_ch;

        // 1. Normalize target
        Scalar z_mean = target.mean();
        Scalar z_std = std::sqrt(
            (target.array() - z_mean).square().sum()
            / static_cast<Scalar>(n_epochs - 1));
        VectorType z = (target.array() - z_mean).matrix();
        if (z_std > Scalar{1e-15}) z /= z_std;

        // 2. Per-epoch covariance → C and Cz
        MatrixType C  = MatrixType::Zero(n_ch, n_ch);
        MatrixType Cz = MatrixType::Zero(n_ch, n_ch);

        for (Eigen::Index e = 0; e < n_epochs; ++e) {
            const auto& X = epochs[static_cast<size_t>(e)];
            MatrixType Xc = X.colwise() - X.rowwise().mean();
            MatrixType cov_e = (Xc * Xc.transpose())
                             / static_cast<Scalar>(Xc.cols());
            C  += cov_e;
            Cz += z(e) * cov_e;
        }
        C  /= static_cast<Scalar>(n_epochs);
        Cz /= static_cast<Scalar>(n_epochs);

        // 3. Generalized eigenvalue problem: Cz w = λ C w
        Eigen::GeneralizedSelfAdjointEigenSolver<MatrixType> solver(Cz, C);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error(
                "SPoC::fit: eigenvalue decomposition failed");
        }

        const VectorType& all_evals = solver.eigenvalues();
        const MatrixType& all_evecs = solver.eigenvectors();

        // 4. Sort by |eigenvalue| descending
        std::vector<int> idx(static_cast<size_t>(n_ch));
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) {
            return std::abs(all_evals(a)) > std::abs(all_evals(b));
        });

        int n_comp = std::min(n_components_, static_cast<int>(n_ch));
        filters_.resize(n_comp, n_ch);
        eigenvalues_.resize(n_comp);

        for (int i = 0; i < n_comp; ++i) {
            int j = idx[static_cast<size_t>(i)];
            eigenvalues_(i) = all_evals(j);
            VectorType w = all_evecs.col(j);
            Scalar norm = w.norm();
            if (norm > Scalar{0}) w /= norm;
            filters_.row(i) = w.transpose();
        }

        // 5. Patterns: A = C W inv(W^T C W)
        MatrixType W = filters_.transpose();
        MatrixType CW = C * W;
        MatrixType WtCW = W.transpose() * CW;
        patterns_ = (CW * WtCW.inverse()).transpose();

        this->fitted_ = true;
        return *this;
    }

    // -- transform ----------------------------------------------------------

    /// @brief Extract log-variance features from epochs.
    ///
    /// @param epochs Vector of epoch matrices (n_channels × n_times each).
    /// @return Feature matrix of shape (n_epochs × n_components).
    [[nodiscard]] MatrixType transform(
        const std::vector<MatrixType>& epochs) const {
        this->check_is_fitted();

        const int n_epochs = static_cast<int>(epochs.size());
        const int n_comp = static_cast<int>(filters_.rows());
        MatrixType features(n_epochs, n_comp);

        for (int e = 0; e < n_epochs; ++e) {
            MatrixType s = filters_ * epochs[static_cast<size_t>(e)];
            for (int c = 0; c < n_comp; ++c) {
                Scalar var = s.row(c).squaredNorm()
                           / static_cast<Scalar>(s.cols());
                features(e, c) = std::log(std::max(var, Scalar{1e-30}));
            }
        }

        return features;
    }

    /// @brief Fit and transform in one step.
    [[nodiscard]] MatrixType fit_transform(
        const std::vector<MatrixType>& epochs,
        const Eigen::Ref<const VectorType>& target) {
        fit(epochs, target);
        return transform(epochs);
    }

private:
    int n_components_;
    MatrixType filters_;
    MatrixType patterns_;
    VectorType eigenvalues_;
};

/// @}

}  // namespace Skigen

#endif  // SKIGEN_DECOMPOSITION_SPOC_H
