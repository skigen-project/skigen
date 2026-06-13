// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors
//
// Entry point for the Skigen parity suite. Each module contributes a
// parity_<module>() function that loads the checked-in scikit-learn reference
// fixtures and asserts agreement within a per-estimator tolerance.

#include <iostream>

#include "parity_common.h"

void parity_core();
void parity_naive_bayes();
void parity_isotonic();
void parity_covariance();
void parity_feature_selection();
void parity_bayesian_linear();
void parity_decomposition();
void parity_neighbors();
void parity_ensemble();
void parity_svm();
void parity_neural_network();
void parity_manifold();
void parity_calibration();
void parity_model_selection();

int main() {
    std::cout << "=== Skigen scikit-learn parity suite ===\n";

    parity_core();
    parity_naive_bayes();
    parity_isotonic();
    parity_covariance();
    parity_feature_selection();
    parity_bayesian_linear();
    parity_decomposition();
    parity_neighbors();
    parity_ensemble();
    parity_svm();
    parity_neural_network();
    parity_manifold();
    parity_calibration();
    parity_model_selection();

    return skigen_parity::summary();
}
