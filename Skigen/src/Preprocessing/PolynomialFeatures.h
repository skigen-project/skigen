// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#ifndef SKIGEN_PREPROCESSING_POLYNOMIAL_FEATURES_H
#define SKIGEN_PREPROCESSING_POLYNOMIAL_FEATURES_H

#include "../Core/Base.h"
#include "../Core/Validation.h"

#include <Eigen/Core>
#include <stdexcept>
#include <vector>

namespace Skigen {

/// @defgroup Algo_PolynomialFeatures PolynomialFeatures
/// @ingroup Preprocessing
/// @brief Generate polynomial and interaction features.
/// @{

/// @brief Generate polynomial and interaction features.
///
/// Generate a new feature matrix consisting of all polynomial combinations
/// of the features with degree less than or equal to the specified degree.
/// For example, if an input sample is two dimensional and of the form
/// `[a, b]`, the degree-2 polynomial features are `[1, a, b, a², ab, b²]`.
///
/// Mirrors
/// [sklearn.preprocessing.PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html).
///
/// ### Parameters (constructor)
///
/// | Parameter | Type | Default | Description |
/// |-----------|------|---------|-------------|
/// | `degree` | `int` | `2` | The maximum degree of the polynomial features. |
/// | `include_bias` | `bool` | `true` | If `true`, include a bias column of all ones. |
/// | `interaction_only` | `bool` | `false` | If `true`, only interaction features are produced (no `x^2`, `x^3`, etc.). |
///
/// ### Attributes (after fitting)
///
/// | Accessor | Type | Description |
/// |----------|------|-------------|
/// | `n_output_features()` | `IndexType` | Number of output features after transformation. |
///
/// ### Notes
///
/// `inverse_transform()` is not supported.
///
/// ### Limitations relative to scikit-learn
///
/// The following scikit-learn constructor
///   parameters are not honoured: `order`.
///   The following sklearn fitted attributes are not exposed:
///   `powers_`, `n_features_in_`, `feature_names_in_`.
///   `get_feature_names_out()` is not implemented.
///
/// ### Examples
///
/// @snippet polynomial_features.cpp example_polynomial_features
template <typename Scalar = double>
class PolynomialFeatures
    : public Transformer<PolynomialFeatures<Scalar>, Scalar> {
public:
    using Base = Transformer<PolynomialFeatures<Scalar>, Scalar>;
    using typename Base::MatrixType;
    using typename Base::VectorType;
    using typename Base::RowVectorType;
    using typename Base::IndexType;

    /// @brief Construct a PolynomialFeatures transformer.
    ///
    /// @param degree Maximum polynomial degree (`int`, default `2`).
    /// @param include_bias If `true`, include a bias column of ones (`bool`, default `true`).
    /// @param interaction_only If `true`, only interaction features are produced (`bool`, default `false`).
    explicit PolynomialFeatures(int degree = 2, bool include_bias = true,
                                bool interaction_only = false)
        : degree_(degree), include_bias_(include_bias),
          interaction_only_(interaction_only) {}

    /// @brief Maximum polynomial degree.
    [[nodiscard]] int degree() const noexcept { return degree_; }
    /// @brief Number of output features after transformation.
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] Eigen::Index n_output_features() const {
        this->check_is_fitted();
        return n_output_features_;
    }

    /// @brief Compute the powers matrix for later transformation.
    ///
    /// @param X Training data of shape (n_samples, n_features).
    /// @return Reference to the fitted transformer (`*this`).
    PolynomialFeatures& fit_impl(const Eigen::Ref<const MatrixType>& X) {
        internal::check_non_empty(X);
        this->n_features_in_ = X.cols();

        // Build the powers matrix that describes each output feature
        build_powers(X.cols());

        this->fitted_ = true;
        return *this;
    }

    /// @brief Transform data to polynomial features.
    ///
    /// @param X Data matrix of shape (n_samples, n_features).
    /// @return Transformed matrix of shape (n_samples, n_output_features).
    /// @throws std::runtime_error if the model has not been fitted.
    [[nodiscard]] MatrixType transform_impl(
        const Eigen::Ref<const MatrixType>& X) const {
        const Eigen::Index n = X.rows();
        MatrixType result(n, n_output_features_);

        for (Eigen::Index col = 0; col < n_output_features_; ++col) {
            const auto& pows = powers_[static_cast<std::size_t>(col)];
            result.col(col).setOnes();
            for (Eigen::Index f = 0; f < this->n_features_in_; ++f) {
                int p = pows[static_cast<std::size_t>(f)];
                for (int k = 0; k < p; ++k) {
                    result.col(col).array() *= X.col(f).array();
                }
            }
        }

        return result;
    }

    /// @brief Not supported — polynomial feature generation is not reversible.
    ///
    /// @throws std::runtime_error Always throws.
    [[nodiscard]] MatrixType inverse_transform_impl(
        const Eigen::Ref<const MatrixType>& /*X*/) const {
        throw std::runtime_error(
            "PolynomialFeatures does not support inverse_transform.");
    }

private:
    int degree_;
    bool include_bias_;
    bool interaction_only_;
    Eigen::Index n_output_features_ = 0;

    // Each element is a vector of per-feature powers
    std::vector<std::vector<int>> powers_;

    void build_powers(Eigen::Index n_features) {
        powers_.clear();

        // Generate all combinations of powers that sum to <= degree_
        std::vector<int> current(static_cast<std::size_t>(n_features), 0);
        int start_degree = include_bias_ ? 0 : 1;

        for (int d = start_degree; d <= degree_; ++d) {
            generate_combinations(current, 0, d, n_features);
        }

        n_output_features_ = static_cast<Eigen::Index>(powers_.size());
    }

    void generate_combinations(std::vector<int>& current,
                               std::size_t feature_idx,
                               int remaining_degree,
                               Eigen::Index n_features) {
        if (feature_idx == static_cast<std::size_t>(n_features)) {
            if (remaining_degree == 0) {
                powers_.push_back(current);
            }
            return;
        }

        int max_pow = interaction_only_ ? std::min(1, remaining_degree)
                                        : remaining_degree;

        for (int p = max_pow; p >= 0; --p) {
            current[feature_idx] = p;
            generate_combinations(current, feature_idx + 1,
                                  remaining_degree - p, n_features);
        }
        current[feature_idx] = 0;
    }
};

/// @}

} // namespace Skigen

#endif // SKIGEN_PREPROCESSING_POLYNOMIAL_FEATURES_H
