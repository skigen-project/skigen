// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// label_encoder.cpp — LabelEncoder: encode integer labels as contiguous indices
#include <Skigen/Preprocessing>
#include <Eigen/Core>
#include <iostream>

int main() {
    // Non-contiguous labels
    Eigen::VectorXi labels(8);
    labels << 5, 10, 5, 20, 10, 20, 5, 10;

    std::cout << "Original labels: " << labels.transpose() << "\n\n";

    Skigen::LabelEncoder<int> encoder;
    auto encoded = encoder.fit_transform(labels);

    std::cout << "Encoded:  " << encoded.transpose() << "\n";
    std::cout << "Classes:  ";
    for (auto c : encoder.classes()) std::cout << c << " ";
    std::cout << "\n";
    std::cout << "n_classes: " << encoder.n_classes() << "\n\n";

    // Inverse transform — recover original labels
    auto decoded = encoder.inverse_transform(encoded);
    std::cout << "Decoded:  " << decoded.transpose() << "\n\n";

    // Verify round-trip
    bool match = (labels == decoded);
    std::cout << "Round-trip match: " << (match ? "yes" : "no") << "\n";

    return 0;
}
