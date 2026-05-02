// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_DECOMPOSITION_SSD_H
#define SKIGEN_DECOMPOSITION_SSD_H

#include "../Core/Base.h"
#include "../Core/Validation.h"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace Skigen {

/// @defgroup Algo_SSD SSD
/// @ingroup Decomposition
/// @brief Spatio-Spectral Decomposition.
/// @{

/// @brief Spatio-Spectral Decomposition (SSD).
///
/// Finds spatial filters that maximize signal-band power relative to
/// noise-band power. Used in rhythmic brain activity analysis.
///
/// ### Algorithm
///
/// 1. Bandpass-filter the data into signal and noise bands.
/// 2. Compute covariance matrices @f$C_\text{signal}@f$ and
///    @f$C_\text{noise}@f$.
/// 3. Regularize @f$C_\text{noise}@f$ with parameter @f$\alpha@f$.
/// 4. Solve the generalized eigenvalue problem
///    @f$C_\text{signal} w = \lambda C_\text{noise} w@f$.
/// 5. Select top eigenvectors as spatial filters.
///
/// Mirrors
/// [mne.decoding.SSD](https://mne.tools/stable/generated/mne.decoding.SSD.html).
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `n_components` | `int` | 6 | Number of SSD components. |
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type | Description |
/// |----------|------|-------------|
/// | `filters()` | `MatrixType` | Spatial filters (n_components × n_channels). |
/// | `patterns()` | `MatrixType` | Spatial patterns (n_components × n_channels). |
/// | `eigenvalues()` | `VectorType` | Eigenvalues (signal/noise power ratio). |
///
template <typename Scalar = double>
class SSD
    : public Estimator<SSD<Scalar>, Scalar> {
public:
    using Base = Estimator<SSD<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;
    using typename Base::ScalarType;

    /// @brief Construct an SSD estimator.
    explicit SSD(int n_components = 6)
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

    /// @brief Eigenvalues (signal/noise power ratio).
    [[nodiscard]] const VectorType& eigenvalues() const {
        this->check_is_fitted();
        return eigenvalues_;
    }

    // -- fit ----------------------------------------------------------------

    /// @brief Fit SSD from continuous data.
    ///
    /// @param data       Continuous data matrix (n_channels × n_times).
    /// @param sfreq      Sampling frequency in Hz.
    /// @param signal_low  Lower edge of the signal band (Hz).
    /// @param signal_high Upper edge of the signal band (Hz).
    /// @param noise_low   Lower edge of the noise band (Hz).
    /// @param noise_high  Upper edge of the noise band (Hz).
    /// @param reg_param   Regularization parameter for noise covariance.
    /// @return Reference to the fitted estimator.
    SSD& fit(const Eigen::Ref<const MatrixType>& data,
             Scalar sfreq,
             Scalar signal_low, Scalar signal_high,
             Scalar noise_low, Scalar noise_high,
             Scalar reg_param = Scalar{0.05}) {
        const auto n_ch = data.rows();
        const auto n_times = data.cols();
        this->n_features_in_ = n_ch;

        if (n_ch < 2 || n_times < 2) {
            throw std::invalid_argument(
                "SSD::fit: data must have >= 2 channels and >= 2 time points");
        }

        int n_comp = std::min(n_components_, static_cast<int>(n_ch));

        // Mean-center
        MatrixType centered = data.colwise() - data.rowwise().mean();

        // Bandpass filter for signal and noise bands
        MatrixType data_signal = bandpass_filter(
            centered, sfreq, signal_low, signal_high);
        MatrixType data_noise = bandpass_filter(
            centered, sfreq, noise_low, noise_high);

        // Covariance matrices
        MatrixType cov_signal = (data_signal * data_signal.transpose())
                              / static_cast<Scalar>(n_times - 1);
        MatrixType cov_noise  = (data_noise * data_noise.transpose())
                              / static_cast<Scalar>(n_times - 1);

        // Regularize noise covariance
        cov_noise += reg_param * cov_noise.trace()
                   / static_cast<Scalar>(n_ch)
                   * MatrixType::Identity(n_ch, n_ch);

        // Generalized eigenvalue problem: C_signal w = λ C_noise w
        Eigen::GeneralizedSelfAdjointEigenSolver<MatrixType> ges(
            cov_signal, cov_noise);
        if (ges.info() != Eigen::Success) {
            throw std::runtime_error(
                "SSD::fit: eigenvalue decomposition failed");
        }

        // Eigenvalues ascending → reverse for descending
        VectorType evals = ges.eigenvalues().reverse();
        MatrixType evecs = ges.eigenvectors().rowwise().reverse();

        eigenvalues_ = evals.head(n_comp);
        filters_ = evecs.leftCols(n_comp).transpose();  // (n_comp × n_ch)

        // Patterns: A = C W inv(W^T C W)
        MatrixType cov_full = (centered * centered.transpose())
                            / static_cast<Scalar>(n_times - 1);
        MatrixType WtCW = filters_ * cov_full * filters_.transpose();
        patterns_ = (cov_full * filters_.transpose()
                     * WtCW.inverse()).transpose();

        this->fitted_ = true;
        return *this;
    }

    // -- transform ----------------------------------------------------------

    /// @brief Apply spatial filters to data.
    ///
    /// @param data Continuous data (n_channels × n_times).
    /// @return Filtered data (n_components × n_times).
    [[nodiscard]] MatrixType transform(
        const Eigen::Ref<const MatrixType>& data) const {
        this->check_is_fitted();
        return filters_ * data;
    }

    /// @brief Fit and transform in one step.
    [[nodiscard]] MatrixType fit_transform(
        const Eigen::Ref<const MatrixType>& data,
        Scalar sfreq,
        Scalar signal_low, Scalar signal_high,
        Scalar noise_low, Scalar noise_high,
        Scalar reg_param = Scalar{0.05}) {
        fit(data, sfreq, signal_low, signal_high,
            noise_low, noise_high, reg_param);
        return transform(data);
    }

private:
    int n_components_;
    MatrixType filters_;
    MatrixType patterns_;
    VectorType eigenvalues_;

    /// @brief Windowed-sinc FIR bandpass filter (zero-phase).
    static MatrixType bandpass_filter(
        const MatrixType& data,
        Scalar sfreq, Scalar low_freq, Scalar high_freq) {
        const auto n_ch = data.rows();
        const auto n_times = data.cols();

        int filter_order = std::min(
            static_cast<int>(std::round(Scalar{3} * sfreq / low_freq)),
            static_cast<int>(n_times) - 1);
        if (filter_order % 2 != 0) ++filter_order;
        filter_order = std::min(filter_order, 128);

        const int half = filter_order / 2;
        const Scalar pi = Scalar{M_PI};

        // Windowed-sinc coefficients
        VectorType h(filter_order + 1);
        Scalar w_low  = Scalar{2} * pi * low_freq  / sfreq;
        Scalar w_high = Scalar{2} * pi * high_freq / sfreq;

        for (int i = 0; i <= filter_order; ++i) {
            int n = i - half;
            if (n == 0) {
                h(i) = (w_high - w_low) / pi;
            } else {
                h(i) = (std::sin(w_high * static_cast<Scalar>(n))
                       - std::sin(w_low * static_cast<Scalar>(n)))
                      / (pi * static_cast<Scalar>(n));
            }
            // Hamming window
            h(i) *= Scalar{0.54}
                   - Scalar{0.46} * std::cos(
                       Scalar{2} * pi * static_cast<Scalar>(i)
                       / static_cast<Scalar>(filter_order));
        }

        // Normalize to unit gain at center frequency
        Scalar center = (low_freq + high_freq) / Scalar{2};
        Scalar w_center = Scalar{2} * pi * center / sfreq;
        Scalar gain = Scalar{0};
        for (int i = 0; i <= filter_order; ++i) {
            gain += h(i) * std::cos(
                w_center * static_cast<Scalar>(i - half));
        }
        if (std::abs(gain) > Scalar{1e-12}) h /= std::abs(gain);

        // Forward + reverse convolution (zero-phase)
        MatrixType result(n_ch, n_times);

        for (Eigen::Index ch = 0; ch < n_ch; ++ch) {
            VectorType forward(n_times);
            for (Eigen::Index t = 0; t < n_times; ++t) {
                Scalar sum = Scalar{0};
                for (int k = 0; k <= filter_order; ++k) {
                    auto idx = t - k;
                    if (idx >= 0 && idx < n_times)
                        sum += h(k) * data(ch, idx);
                }
                forward(t) = sum;
            }

            VectorType backward(n_times);
            for (Eigen::Index t = n_times - 1; t >= 0; --t) {
                Scalar sum = Scalar{0};
                for (int k = 0; k <= filter_order; ++k) {
                    auto idx = t + k;
                    if (idx >= 0 && idx < n_times)
                        sum += h(k) * forward(idx);
                }
                backward(t) = sum;
            }

            result.row(ch) = backward.transpose();
        }

        return result;
    }
};

/// @}

}  // namespace Skigen

#endif  // SKIGEN_DECOMPOSITION_SSD_H
