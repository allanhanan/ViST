#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <functional>
#include <random>


// Matrix multiplication (A * B)
inline std::vector<std::vector<float>> matmul(const std::vector<std::vector<float>>& A, 
                                                const std::vector<std::vector<float>>& B) {
    int rowsA = A.size(), colsA = A[0].size();
    int rowsB = B.size(), colsB = B[0].size();
    
    if (colsA != rowsB) {
        throw std::invalid_argument("Matrix dimensions are not compatible for multiplication");
    }

    std::vector<std::vector<float>> result(rowsA, std::vector<float>(colsB, 0.0f));

    // Optimized matrix multiplication using cache-friendly memory access
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < colsA; ++k) {
                sum += A[i][k] * B[k][j];
            }
            result[i][j] = sum;
        }
    }
    return result;
}
 

inline std::vector<std::vector<float>> scalar_multiply(const std::vector<std::vector<float>>& matrix, float scalar) {
    std::vector<std::vector<float>> output = matrix;
    for (auto& row : output) {
        for (auto& val : row) {
            val *= scalar;
        }
    }
    return output;
}


// Softmax function for a matrix (applies softmax to each row)
inline std::vector<std::vector<float>> softmax(const std::vector<std::vector<float>>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    std::vector<std::vector<float>> result(rows, std::vector<float>(cols));

    // Softmax calculation
    for (int i = 0; i < rows; ++i) {
        float row_max = *std::max_element(matrix[i].begin(), matrix[i].end());
        float sum_exp = 0.0f;
        for (int j = 0; j < cols; ++j) {
            sum_exp += std::exp(matrix[i][j] - row_max);
        }
        for (int j = 0; j < cols; ++j) {
            result[i][j] = std::exp(matrix[i][j] - row_max) / sum_exp;
        }
    }
    return result;
}

// Matrix transpose
inline std::vector<std::vector<float>> transpose(const std::vector<std::vector<float>>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    std::vector<std::vector<float>> transposed(cols, std::vector<float>(rows));

    // Optimized in-place transpose
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            transposed[j][i] = matrix[i][j];
        }
    }
    return transposed;
}

// Linear transformation (A * W + b)
inline std::vector<std::vector<float>> linear_transform(const std::vector<std::vector<float>>& input, 
                                                          const std::vector<std::vector<float>>& weights, 
                                                          const std::vector<float>& bias) {
    int rows = input.size();
    int cols = weights[0].size();
    std::vector<std::vector<float>> output(rows, std::vector<float>(cols));

    // Linear transformation with in-place computation
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            output[i][j] = bias[j];
            for (int k = 0; k < weights.size(); ++k) {
                output[i][j] += input[i][k] * weights[k][j];
            }
        }
    }
    return output;
}

// Matrix addition (A + B)
inline std::vector<std::vector<float>> add_matrices(const std::vector<std::vector<float>>& A, 
                                                    const std::vector<std::vector<float>>& B) {
    int rows = A.size();
    int cols = A[0].size();
    std::vector<std::vector<float>> result(rows, std::vector<float>(cols));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i][j] = A[i][j] + B[i][j];
        }
    }
    return result;
}

inline std::vector<std::vector<float>> layer_norm(const std::vector<std::vector<float>>& input, 
                                                   const std::vector<float>& gamma, 
                                                   const std::vector<float>& beta, 
                                                   float epsilon = 1e-5f) {
    int rows = input.size();
    int cols = input[0].size();
    std::vector<std::vector<float>> output(rows, std::vector<float>(cols));

    // Layer normalization in a single loop for mean and variance
    for (int i = 0; i < rows; ++i) {
        float mean = 0.0f, variance = 0.0f;
        
        // Compute mean
        for (int j = 0; j < cols; ++j) {
            mean += input[i][j];
        }
        mean /= cols;

        // Compute variance
        for (int j = 0; j < cols; ++j) {
            variance += (input[i][j] - mean) * (input[i][j] - mean);
        }
        variance /= cols;

        // Handle small variance values by adjusting epsilon
        variance = (variance < epsilon) ? epsilon : variance;

        // Apply normalization
        for (int j = 0; j < cols; ++j) {
            output[i][j] = ((input[i][j] - mean) / std::sqrt(variance + epsilon)) * gamma[j] + beta[j];
            if (std::isnan(output[i][j]) || std::isinf(output[i][j])) {
                std::cerr << "[ERROR] NaN or Inf detected in output of layer normalization!" << std::endl;
                return {};  // Return empty to signal error
            }
        }
    }
    return output;
}

// ReLU activation
inline std::vector<std::vector<float>> relu(const std::vector<std::vector<float>>& input) {
    std::vector<std::vector<float>> output = input;
    for (auto& row : output) {
        for (auto& val : row) {
            val = std::max(0.0f, val);
        }
    }
    return output;
}

// GELU activation
inline std::vector<std::vector<float>> gelu(const std::vector<std::vector<float>>& input) {
    std::vector<std::vector<float>> output = input;
    for (auto& row : output) {
        for (auto& val : row) {
            val = 0.5f * val * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * (val + 0.044715f * val * val * val)));
        }
    }
    return output;
}

inline std::vector<std::vector<float>> random_matrix(int rows, int cols, float min_val = -0.1f, float max_val = 0.1f) {
    std::vector<std::vector<float>> matrix(rows, std::vector<float>(cols));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);

    // In-place random matrix generation
    for (auto& row : matrix) {
        for (auto& val : row) {
            val = dis(gen);
        }
    }
    return matrix;
}

// Random vector generator.
inline std::vector<float> random_vector(int size, float min_val = -0.1f, float max_val = 0.1f) {
    std::vector<float> vec(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);
    
    // In-place random vector generation
    for (auto& val : vec) {
        val = dis(gen);
    }
    return vec;
}

// Add bias to a matrix.
inline std::vector<std::vector<float>> add_bias(const std::vector<std::vector<float>>& input, 
                                                const std::vector<float>& bias) {
    int rows = input.size();
    int cols = input[0].size();
    std::vector<std::vector<float>> output = input;
    
    // In-place bias addition
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            output[i][j] += bias[j];
        }
    }
    return output;
}

// Softmax function for a single vector.
inline std::vector<float> softmax_vector(const std::vector<float>& logits) {
    std::vector<float> exps;
    float max_logit = *std::max_element(logits.begin(), logits.end());
    float sum = 0.0f;

    for (float logit : logits) {
        float e = std::exp(logit - max_logit);  // Subtract max(logit) for stability
        exps.push_back(e);
        sum += e;
    }
    for (auto &val : exps) {
        val /= sum;
    }
    return exps;
}


inline float calculate_loss(const std::vector<float>& logits, int correct_class) {
    if (correct_class < 0 || correct_class >= logits.size()) {
        throw std::invalid_argument("Invalid correct class index");
    }
    auto probabilities = softmax_vector(logits);
    float loss = -std::log(probabilities[correct_class] + 1e-9f); // epsilon for numerical stability
    return loss;
}

bool is_nan_or_inf(float value) {
    return std::isnan(value) || std::isinf(value);
}

#endif // UTILS_HPP
