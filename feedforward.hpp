#ifndef FEEDFORWARD_HPP
#define FEEDFORWARD_HPP

#include <vector>
#include "utils.hpp"

// Inline function to implement the feedforward network.
inline std::vector<std::vector<float>> feedforward_network(
    const std::vector<std::vector<float>>& input,
    const std::vector<std::vector<float>>& W1,  // Weights for the first layer
    const std::vector<float>& bias1,             // Bias for the first layer
    const std::vector<std::vector<float>>& W2,    // Weights for the second layer
    const std::vector<float>& bias2              // Bias for the second layer
) {
    // First linear transformation (expansion)
    auto hidden = linear_transform(input, W1, bias1);
    // Apply non-linearity (GELU)
    hidden = gelu(hidden);
    // Second linear transformation (projection)
    auto output = linear_transform(hidden, W2, bias2);
    return output;
}

#endif // FEEDFORWARD_HPP
