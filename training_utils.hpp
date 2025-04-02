#ifndef TRAINING_UTILS_HPP
#define TRAINING_UTILS_HPP

#include <vector>
#include <cstdlib>
#include <iostream>
#include <cmath>

// Dropout function for a matrix
inline std::vector<std::vector<float>> dropout(const std::vector<std::vector<float>>& input, float drop_prob) {
    int rows = input.size();
    int cols = input[0].size();
    std::vector<std::vector<float>> output = input;

    float scale_factor = 1.0f / (1.0f - drop_prob);
    scale_factor = std::min(scale_factor, 10.0f);  // Avoid excessively large scaling

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float r = static_cast<float>(rand()) / RAND_MAX;
            if (r < drop_prob) {
                output[i][j] = 0.0f;
            } else {
                output[i][j] *= scale_factor;
            }
        }
    }
    return output;
}


// Early stopping structure to monitor validation loss
struct EarlyStopping {
    int patience;  // Number of epochs without improvement before stopping
    int counter;
    float best_loss;
    bool stop;

    EarlyStopping(int p) : patience(p), counter(0), best_loss(1e9), stop(false) {}

    // Call this function at the end of each epoch with the current validation loss
    void update(float current_loss) {
        if (current_loss < best_loss) {
            best_loss = current_loss;
            counter = 0;
        } else {
            counter++;
            if (counter >= patience) {
                stop = true;
                std::cout << "Early stopping triggered after " << patience << " epochs without improvement." << std::endl;
            }
        }
    }
};

// Loss function
inline float cross_entropy_loss(const std::vector<float>& logits, int true_label) {
    if (true_label >= logits.size()) {
        throw std::invalid_argument("True label is out of range.");
    }

    // Prevent log(0) by clamping logits[true_label] to epsilon range
    float epsilon = 1e-9f;
    float logit_value = std::max(logits[true_label], epsilon);  // Clamp logit value to avoid log(0)
    float loss = -std::log(logit_value);
    return loss;
}


inline std::vector<float> cross_entropy_loss_grad(const std::vector<float>& logits, int true_label) {
    std::vector<float> grad(logits.size(), 0.0f);

    float epsilon = 1e-9f;  // Small constant to avoid division by zero
    float logit_value = std::max(logits[true_label], epsilon);  // Clamp the logit value

    grad[true_label] = -1.0f / logit_value;  // Derivative of cross-entropy loss
    return grad;
}

inline void update_classifier_weights(std::vector<std::vector<float>>& weights,
                                      std::vector<float>& biases,
                                      std::vector<float>& grad_logits,  // Remove const here
                                      const std::vector<float>& input,  // 1D input vector
                                      float learning_rate, float max_grad_norm = 1.0f) {
    int num_features = input.size();
    int num_classes = weights[0].size();

    // Apply gradient clipping if needed
    float grad_norm = 0.0f;
    for (float grad : grad_logits) {
        grad_norm += grad * grad;
    }
    grad_norm = std::sqrt(grad_norm);
    if (grad_norm > max_grad_norm) {
        float scaling_factor = max_grad_norm / grad_norm;
        for (float& grad : grad_logits) {
            grad *= scaling_factor;
        }
    }

    // Update weights using the gradient of the loss w.r.t. the logits
    for (int j = 0; j < num_classes; ++j) {
        for (int i = 0; i < num_features; ++i) {
            weights[i][j] -= learning_rate * grad_logits[j] * input[i];
        }
    }

    // Update biases (assuming a bias term for each class)
    for (int j = 0; j < num_classes; ++j) {
        biases[j] -= learning_rate * grad_logits[j];
    }
}


// Backpropagation for Transformer Block Classifier
inline void backpropagate_classifier(std::vector<std::vector<float>>& classifier_weights,
    std::vector<float>& classifier_bias,
    const std::vector<float>& logits,
    int true_label,
    float learning_rate) {
    // Compute the gradient of cross-entropy loss with respect to the logits
    std::vector<float> grad_logits = cross_entropy_loss_grad(logits, true_label);

    // Directly pass the logits as a 1D vector (no need to wrap it in a 2D vector)
    update_classifier_weights(classifier_weights, classifier_bias, grad_logits, {logits}, learning_rate);
}



// --- Backpropagation for the Vision Transformer Model ---
inline void backpropagate_vit(std::vector<float>& vit_logits,
                              int true_label,
                              std::vector<std::vector<float>>& classifier_weights,
                              std::vector<float>& classifier_bias,
                              float learning_rate) {
    // Backpropagate through the classifier
    backpropagate_classifier(classifier_weights, classifier_bias, vit_logits, true_label, learning_rate);
}


#endif // TRAINING_UTILS_HPP
