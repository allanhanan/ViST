#ifndef OPTIMIZERS_HPP
#define OPTIMIZERS_HPP
#include <vector>
#include <cmath>

// Function to perform Adam update for a matrix parameter
// m and v must have the same shape as param and are updated in-place
// t is the current time step (starting at 1)
inline void adam_update_matrix(std::vector<std::vector<float>> &param,
                               const std::vector<std::vector<float>> &grad,
                               std::vector<std::vector<float>> &m,
                               std::vector<std::vector<float>> &v,
                               float learning_rate,
                               float beta1,
                               float beta2,
                               float epsilon,
                               int t,
                               float l1 = 0.0f,
                               float l2 = 0.0f) {
    // Get the number of rows in the matrix
    int rows = param.size();
    // Get the number of columns in the matrix
    int cols = param[0].size();
    
    // Precompute bias-correction factors for m and v
    float beta1_t = std::pow(beta1, t);
    float beta2_t = std::pow(beta2, t);
    float m_hat_correction = 1.0f / (1.0f - beta1_t);
    float v_hat_correction = 1.0f / (1.0f - beta2_t);
    
    // Loop through the matrix elements
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // Update biased first moment estimate
            m[i][j] = beta1 * m[i][j] + (1 - beta1) * grad[i][j];
            
            // Update biased second moment estimate
            v[i][j] = beta2 * v[i][j] + (1 - beta2) * grad[i][j] * grad[i][j];
            
            // Bias-correction for m and v
            float m_hat = m[i][j] * m_hat_correction;
            float v_hat = v[i][j] * v_hat_correction;
            
            // Regularization terms (only if l1 or l2 are non-zero)
            float reg = 0.0f;
            if (l1 != 0.0f || l2 != 0.0f) {
                reg = l1 * ((param[i][j] >= 0) ? 1.0f : -1.0f) + l2 * 2.0f * param[i][j];
            }
            
            // Update parameter
            param[i][j] -= learning_rate * (m_hat / (std::sqrt(v_hat) + epsilon) + reg);
        }
    }
}

// Function to perform Adam update for a vector parameter
inline void adam_update_vector(std::vector<float> &param,
                               const std::vector<float> &grad,
                               std::vector<float> &m,
                               std::vector<float> &v,
                               float learning_rate,
                               float beta1,
                               float beta2,
                               float epsilon,
                               int t,
                               float l1 = 0.0f,
                               float l2 = 0.0f) {
    // Get the size of the vector
    int size = param.size();
    
    // Precompute bias-correction factors for m and v
    float beta1_t = std::pow(beta1, t);
    float beta2_t = std::pow(beta2, t);
    float m_hat_correction = 1.0f / (1.0f - beta1_t);
    float v_hat_correction = 1.0f / (1.0f - beta2_t);
    
    // Loop through the vector elements
    for (int i = 0; i < size; ++i) {
        // Update biased first moment estimate
        m[i] = beta1 * m[i] + (1 - beta1) * grad[i];
        
        // Update biased second moment estimate
        v[i] = beta2 * v[i] + (1 - beta2) * grad[i] * grad[i];
        
        // Bias-correction for m and v
        float m_hat = m[i] * m_hat_correction;
        float v_hat = v[i] * v_hat_correction;
        
        // Regularization terms (only if l1 or l2 are non-zero)
        float reg = 0.0f;
        if (l1 != 0.0f || l2 != 0.0f) {
            reg = l1 * ((param[i] >= 0) ? 1.0f : -1.0f) + l2 * 2.0f * param[i];
        }
        
        // Update parameter
        param[i] -= learning_rate * (m_hat / (std::sqrt(v_hat) + epsilon) + reg);
    }
}
#endif // OPTIMIZERS_HPP