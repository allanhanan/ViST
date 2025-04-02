#ifndef GRADIENTS_HPP
#define GRADIENTS_HPP

#include <vector>
#include <cmath>
#include <stdexcept>
#include "utils.hpp"

// Gradient for Linear Transformation
// Forward: Y = X * W + b
// Inputs:
//   X          : [batch_size, in_dim]
//   W          : [in_dim, out_dim]
//   b          : [out_dim]
//   grad_Y     : [batch_size, out_dim] (gradient of loss with respect to Y)
// Outputs (by reference):
//   grad_X     : [batch_size, in_dim]
//   grad_W     : [in_dim, out_dim]
//   grad_b     : [out_dim]
inline void backprop_linear_transform(
    const std::vector<std::vector<float>>& X,  // Input data matrix [batch_size, in_dim]
    const std::vector<std::vector<float>>& W,  // Weight matrix for linear transformation [in_dim, out_dim]
    const std::vector<std::vector<float>>& grad_Y,  // Gradient of loss with respect to Y [batch_size, out_dim]
    std::vector<std::vector<float>>& grad_X,  // Gradient of loss with respect to X [batch_size, in_dim] (output)
    std::vector<std::vector<float>>& grad_W,  // Gradient of loss with respect to W [in_dim, out_dim] (output)
    std::vector<float>& grad_b   // Gradient of loss with respect to b [out_dim] (output)

) {
    int batch_size = X.size();
    int in_dim = X[0].size();
    int out_dim = W[0].size();

    // Initialize gradients to zero
    std::fill(grad_b.begin(), grad_b.end(), 0.0f);  // Reset grad_b to zero
    std::fill(grad_W.begin(), grad_W.end(), std::vector<float>(out_dim, 0.0f));  // Reset grad_W to zero
    std::fill(grad_X.begin(), grad_X.end(), std::vector<float>(in_dim, 0.0f));  // Reset grad_X to zero

    // Compute the gradient of the bias term by summing over the batch
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < out_dim; ++j) {
            grad_b[j] += grad_Y[i][j];
        }
    }

    // Compute the gradient of the weight matrix by multiplying the input transposed with the gradient of Y
    // grad_W: X^T * grad_Y.
    std::vector<std::vector<float>> X_T = transpose(X);
    grad_W = matmul(X_T, grad_Y);

    // Compute the gradient of the input by multiplying the gradient of Y with the transposed weight matrix
    //grad_X: grad_Y * W^T.
    std::vector<std::vector<float>> W_T = transpose(W);  // Transpose W only once
    grad_X = matmul(grad_Y, W_T);  // Matrix multiplication
}

// Gradient for GELU Activation
// GELU(x) = 0.5 * x * [1 + tanh( sqrt(2/π) * (x + 0.044715 * x^3) ) ]
// Its derivative is computed using the chain rule.
inline float gelu_derivative(float x) {
    const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);  // Precomputed constant
    float x3 = x * x * x;  // Cube of input
    float tanh_arg = sqrt_2_over_pi * (x + 0.044715f * x3);  // Argument for the hyperbolic tangent function
    float tanh_val = std::tanh(tanh_arg);  // Hyperbolic tangent value
    float sech2 = 1.0f - tanh_val * tanh_val;  // derivative of tanh is 1 - tanh^2
    float left = 0.5f * (1.0f + tanh_val);
    float right = 0.5f * x * sech2 * sqrt_2_over_pi * (1.0f + 3.0f * 0.044715f * x);
    return left + right;
}

// Given the input to GELU (before applying GELU), compute the elementwise derivative.
inline void gelu_gradient(const std::vector<std::vector<float>>& input, std::vector<std::vector<float>>& grad) {
    int rows = input.size();
    int cols = input[0].size();
    
    // Resize grad to match the input size if necessary
    grad.resize(rows, std::vector<float>(cols, 0.0f));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            grad[i][j] = gelu_derivative(input[i][j]);
        }
    }
}

#endif // GRADIENTS_HPP
