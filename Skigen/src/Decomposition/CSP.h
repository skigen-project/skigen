// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_DECOMPOSITION_CSP_H
#define SKIGEN_DECOMPOSITION_CSP_H

#include "../Core/Base.h"
#include "../Core/Validation.h"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace Skigen {

/// @defgroup Algo_CSP CSP
/// @ingroup Decomposition
/// @brief Common Spatial Patterns for binary classification.
/// @{

/// @brief Common Spatial Patterns (CSP) spatial filter.
///
/// Finds spatial filters that maximize variance for one class while
/// minimizing it for the other. Widely used in Brain-Computer Interface
/// (BCI) research for EEG/MEG motor imagery classification.
///
/// ### Algorithm
///
/// 1. Compute per-class average covariance matrices @f$C_1, C_2@f$.
/// 2. Whiten via eigendecomposition of the composite covariance
///    @f$C = C_1 + C_2@f$.
/// 3. Eigendecompose the whitened class-1 covariance.
/// 4. Select the first and last eigenvectors as spatial filters
///    (maximizing class-1 and class-2 variance, respectively).
///
/// Mirrors
/// [mne.decoding.CSP](https://mne.tools/stable/generated/mne.decoding.CSP.html).
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `n_components` | `int` | 6 | Number of CSP components (split evenly between classes). |
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type | Description |
/// |----------|------|-------------|
/// | `filters()` | `MatrixType` | Spatial filters (n_components × n_channels). |
/// | `patterns()` | `MatrixType` | Spatial patterns (n_channels × n_components). |
/// | `eigenvalues()` | `VectorType` | Variance ratios for selected components. |
///
/// ### Notes
///
/// Input epochs have shape (n_channels × n_times) each. This matches
/// the mne-cpp convention. The feature matrix returned by `transform()`
/// has shape (n_epochs × n_components) with log-variance features.
///
template <typename Scalar = double>
class CSP
    : public Estimator<CSP<Scalar>, Scalar> {
public:
    using Base = Estimator<CSP<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;
    using typename Base::ScalarType;

    /// @brief Construct a CSP estimator.
    ///
    /// @param n_components Number of spatial filter components.
    explicit CSP(int n_components = 6)
        : n_components_(n_components) {}

    // -- Accessors ----------------------------------------------------------

    /// @brief Spatial filters of shape (n_components × n_channels).
    [[nodiscard]] const MatrixType& filters() const {
        this->check_is_fitted();
        return filters_;
    }

    /// @brief Spatial patterns of shape (n_channels × n_components).
    [[nodiscard]] const MatrixType& patterns() const {
        this->check_is_fitted();
        return patterns_;
    }

    /// @brief Eigenvalues / variance ratios for selected components.
    [[nodiscard]] const VectorType& eigenvalues() const {
        this->check_is_fitted();
        return eigenvalues_;
    }

    // -- fit ----------------------------------------------------------------

    /// @brief Fit CSP from two sets of epoch covariance matrices.
    ///
    /// Each epoch is a matrix of shape (n_channels × n_times).
    ///
    /// @param epochs1 Epochs for class 1.
    /// @param epochs2 Epochs for class 2.
    /// @return Reference to the fitted estimator.
    CSP& fit(const std::vector<MatrixType>& epochs1,
             const std::vector<MatrixType>& epochs2) {
        if (epochs1.empty() || epochs2.empty()) {
            throw std::invalid_argument(
                "CSP::fit: both epoch sets must be non-empty");
        }

        const auto n_ch = epochs1[0].rows();
        this->n_features_in_ = n_ch;

        // Average covariance for each class
        MatrixType cov1 = MatrixType::Zero(n_ch, n_ch);
        for (const auto& epoch : epochs1) {
            MatrixType centered = epoch.colwise() - epoch.rowwise().mean();
            cov1 += centered * centered.transpose()
                    / static_cast<Scalar>(epoch.cols() - 1);
        }
        cov1 /= static_cast<Scalar>(epochs1.size());

        MatrixType cov2 = MatrixType::Zero(n_ch, n_ch);
        for (const auto& epoch : epochs2) {
            MatrixType centered = epoch.colwise() - epoch.rowwise().mean();
            cov2 += centered * centered.transpose()
                    / static_cast<Scalar>(epoch.cols() - 1);
        }
        cov2 /= static_cast<Scalar>(epochs2.size());

        // Composite covariance
        MatrixType cov_comp = cov1 + cov2;

        // Whitening: W = D^{-1/2} U^T
        Eigen::SelfAdjointEigenSolver<MatrixType> eig_comp(cov_comp);
        VectorType d = eig_comp.eigenvalues();
        MatrixType U = eig_comp.eigenvectors();

        const Scalar d_min = d.maxCoeff() * Scalar{1e-10};
        for (Eigen::Index i = 0; i < d.size(); ++i) {
            if (d(i) < d_min) d(i) = d_min;
        }

        MatrixType W = d.array().sqrt().inverse().matrix().asDiagonal()
                     * U.transpose();

        // Whiten class-1 covariance and eigendecompose
        MatrixType S1 = W * cov1 * W.transpose();
        Eigen::SelfAdjointEigenSolver<MatrixType> eig_s1(S1);
        VectorType lambdas = eig_s1.eigenvalues();
        MatrixType B = eig_s1.eigenvectors();

        // All filters in eigenvalue order (ascending)
        MatrixType all_filters = B.transpose() * W;

        // Select first and last components
        int n_per_class = n_components_ / 2;
        int n_total = std::min(n_components_, static_cast<int>(n_ch));
        n_per_class = std::min(n_per_class, static_cast<int>(n_ch) / 2);

        filters_ = MatrixType(n_total, n_ch);
        eigenvalues_ = VectorType(n_total);

        // Bottom n_per_class (maximize class-2 variance)
        for (int i = 0; i < n_per_class; ++i) {
            filters_.row(i) = all_filters.row(i);
            eigenvalues_(i) = lambdas(i);
        }
        // Top (n_total - n_per_class) (maximize class-1 variance)
        for (int i = 0; i < n_total - n_per_class; ++i) {
            filters_.row(n_per_class + i) =
                all_filters.row(static_cast<int>(n_ch) - 1 - i);
            eigenvalues_(n_per_class + i) =
                lambdas(static_cast<int>(n_ch) - 1 - i);
        }

        // Patterns = pinv(filters)^T
        auto svd = filters_.template bdcSvd<Eigen::ComputeThinU | Eigen::ComputeThinV>();
        patterns_ = svd.solve(MatrixType::Identity(n_total, n_total))
                       .transpose();

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
            MatrixType filtered = filters_ * epochs[static_cast<size_t>(e)];
            for (int c = 0; c < n_comp; ++c) {
                Scalar var = filtered.row(c).squaredNorm()
                           / static_cast<Scalar>(filtered.cols());
                features(e, c) = std::log(var);
            }
        }

        // Normalize: subtract per-epoch mean for scale invariance
        for (int e = 0; e < n_epochs; ++e) {
            features.row(e).array() -= features.row(e).mean();
        }

        return features;
    }

    /// @brief Fit and transform in one step.
    [[nodiscard]] MatrixType fit_transform(
        const std::vector<MatrixType>& epochs1,
        const std::vector<MatrixType>& epochs2) {
        fit(epochs1, epochs2);
        std::vector<MatrixType> all;
        all.reserve(epochs1.size() + epochs2.size());
        all.insert(all.end(), epochs1.begin(), epochs1.end());
        all.insert(all.end(), epochs2.begin(), epochs2.end());
        return transform(all);
    }

private:
    int n_components_;
    MatrixType filters_;
    MatrixType patterns_;
    VectorType eigenvalues_;
};

/// @}

}  // namespace Skigen

#endif  // SKIGEN_DECOMPOSITION_CSP_H
