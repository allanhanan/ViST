#ifndef VIT_MODEL_HPP
#define VIT_MODEL_HPP

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include "image_loader.hpp"     
#include "patch_embedding.cpp"      
#include "positional_encoding.cpp"  
#include "utils.hpp"                
#include "multi_head_attention.hpp"  
#include "transformer_block.cpp"     
#include "feedforward.hpp"          
#include "training_utils.hpp"      


inline std::vector<float> linear_classifier(const std::vector<float>& input,
                                       const std::vector<std::vector<float>>& weights,
                                       const std::vector<float>& bias) {
    int input_dim = input.size();
    int num_classes = weights[0].size();
    std::vector<float> output(num_classes, 0.0f);

    // Small epsilon for numerical stability
    const float epsilon = 1e-6f;
    
    for (int j = 0; j < num_classes; ++j) {
        output[j] = bias[j];
        for (int i = 0; i < input_dim; ++i) {
            output[j] += input[i] * weights[i][j];
        }

        // Sanity check for NaN or Inf
        if (std::isnan(output[j]) || std::isinf(output[j])) {
            std::cerr << "[ERROR] NaN or Inf detected in linear classifier output for class " << j << "!" << std::endl;
            output[j] = epsilon;  // Fallback to a small value if NaN or Inf
        }
    }

    // Log classifier output
    std::cout << "[DEBUG] Linear classifier output for input with size " << input_dim
              << " and weights of size [" << input_dim << ", " << num_classes << "]: ";
    for (const auto& o : output) {
        std::cout << o << " ";
    }
    std::cout << std::endl;

    return output;
}

// --- Vision Transformer Model ---
// This defines the full Vision Transformer model architecture for classification.
// It combines patch embedding, positional encoding, [CLS] token, multiple transformer blocks,
// and a final linear classifier
inline std::vector<float> vision_transformer(const std::string &image_path,
                                      int num_classes,
                                      int num_transformer_blocks) {
    std::cout << "[LOG] Starting Vision Transformer model..." << std::endl;
    
    // 1. Load the image.
    std::cout << "[LOG] Loading image from: " << image_path << std::endl;
    int width, height, channels;
    int target_width = 300;
    int target_height = 300;
    auto image_matrix = load_image(image_path, target_width, target_height, width, height, channels);

    if (image_matrix.empty()) {
        throw std::runtime_error("Error: Image loading failed!");
    }

    // Log the actual dimensions of the image matrix
    std::cout << "[LOG] Image loaded successfully! "
            << "Width: " << image_matrix[0].size()        // Width: number of columns in the first row
            << ", Height: " << image_matrix.size()        // Height: number of rows in the matrix
            << ", Channels: " << image_matrix[0][0].size()  // Channels: number of color channels in a single pixel
            << std::endl;

    // 2. Create patch embeddings.
    int patch_size = 16;  // 16x16 patches.
    std::cout << "[LOG] Creating patch embeddings..." << std::endl;
    auto patch_embeddings = create_patch_embedding(image_matrix, patch_size);
    if (patch_embeddings.empty()) {
        throw std::runtime_error("Error: Patch embeddings could not be created!");
    }
    std::cout << "[LOG] Patch embeddings created! Number of patches: " << patch_embeddings.size()
              << ", Patch embedding size: " << patch_embeddings[0].size() << std::endl;

    // 3. Create positional encoding and add it to patch embeddings.
    std::cout << "[LOG] Creating positional encoding..." << std::endl;
    int num_patches = patch_embeddings.size();
    int embedding_dim = patch_embeddings[0].size();  // e.g., 768
    auto pos_encoding = create_positional_encoding(num_patches, embedding_dim);
    auto patches_with_pos = add_positional_encoding(patch_embeddings, pos_encoding);
    std::cout << "[LOG] Positional encoding created and added to patches." << std::endl;

    // 4. Prepend a learnable [CLS] token.
    std::cout << "[LOG] Prepending [CLS] token..." << std::endl;
    std::vector<float> cls_token = random_vector(embedding_dim);  // Learnable token
    std::vector<std::vector<float>> tokens;
    tokens.push_back(cls_token);  // [CLS] token at index 0
    for (const auto &patch : patches_with_pos) {
        tokens.push_back(patch);
    }
    std::cout << "[LOG] [CLS] token prepended. Total tokens: " << tokens.size() << std::endl;

    // 5. Stack Transformer Blocks.
    std::cout << "[LOG] Stacking " << num_transformer_blocks << " transformer blocks..." << std::endl;
    // For residual compatibility, set d_k = d_v = embedding_dim.
    std::vector<TransformerBlock> transformer_blocks;
    for (int i = 0; i < num_transformer_blocks; ++i) {
        // Using 8 attention heads (hyperparameter tooning)
        transformer_blocks.emplace_back(embedding_dim, 8);
    }
    for (int i = 0; i < num_transformer_blocks; ++i) {
        tokens = transformer_blocks[i].forward(tokens);
        std::cout << "[LOG] Transformer block " << (i + 1) << "/" << num_transformer_blocks << " applied." << std::endl;
    }
    std::cout << "[LOG] All transformer blocks applied." << std::endl;

    // 6. Final Layer Normalization
    std::cout << "[LOG] Applying final layer normalization..." << std::endl;
    tokens = layer_norm(tokens, std::vector<float>(embedding_dim, 1.0f),
                                std::vector<float>(embedding_dim, 0.0f));

    // Check for NaN or Inf after layer normalization
    for (const auto& token : tokens) {
        for (const auto& val : token) {
            if (std::isnan(val) || std::isinf(val)) {
                std::cerr << "[ERROR] NaN or Inf detected after layer normalization!" << std::endl;
                return {};  // Return empty to signal error
            }
        }
    }

    // 7. Classification Head.
    std::cout << "[LOG] Applying classification head..." << std::endl;
    // Using the output corresponding to the [CLS] token (first token) for classification.
    std::vector<float> cls_output = tokens[0];  // Shape: [embedding_dim]
    
    // Apply logits clipping to avoid overflow in softmax
    const float logit_clip_value = 20.0f;
    for (auto& val : cls_output) {
        val = std::max(std::min(val, logit_clip_value), -logit_clip_value);
    }

    // Initialize classifier weights: shape [embedding_dim, num_classes]
    auto classifier_weights = random_matrix(embedding_dim, num_classes);
    auto classifier_bias = random_vector(num_classes);
    std::vector<float> logits = linear_classifier(cls_output, classifier_weights, classifier_bias);
    std::cout << "[LOG] Classification head applied." << std::endl;

    std::cout << "[LOG] Vision Transformer model processing complete." << std::endl;
    return logits;  // Raw logits for each class.
}

struct VitModelWrapper {
    int embedding_dim;
    int num_classes;
    int num_blocks;

    std::vector<std::vector<float>> classifier_weights; // [embedding_dim, num_classes]
    std::vector<float> classifier_bias;                 // [num_classes]

    // Gradients for the classifier weights and bias
    std::vector<std::vector<float>> classifier_weights_grad; // Gradients for classifier weights
    std::vector<float> classifier_bias_grad;                // Gradients for classifier bias

    VitModelWrapper(int embed_dim, int n_classes, int n_blocks)
        : embedding_dim(embed_dim), num_classes(n_classes), num_blocks(n_blocks) {
        classifier_weights = random_matrix(embed_dim, num_classes);
        classifier_bias = random_vector(num_classes);

        // Initialize gradients to zero
        classifier_weights_grad = random_matrix(embed_dim, num_classes, 0.0f);
        classifier_bias_grad = random_vector(num_classes, 0.0f);

        std::cout << "[LOG] Classifier weights initialized with shape [" << embed_dim << ", " << n_classes << "]" << std::endl;
        std::cout << "[LOG] Classifier bias initialized with shape [" << n_classes << "]" << std::endl;
    }

    // Forward pass
    std::vector<float> forward(const std::string &image_path) {
        std::vector<float> vit_logits = vision_transformer(image_path, num_classes, num_blocks);
        
        std::cout << "[LOG] Vision Transformer logits: ";
        for (const auto& val : vit_logits) {
            std::cout << val << " ";
        }
        std::cout << std::endl;

        return linear_classifier(vit_logits, classifier_weights, classifier_bias);
    }

    // Compute loss
    float compute_loss(const std::string &image_path, int true_label) {
        std::vector<float> logits = forward(image_path);
        float loss = cross_entropy_loss(logits, true_label);

        std::cout << "[LOG] Loss computed: " << loss << std::endl;
        
        return loss;
    }

    // Backpropagation: Update model parameters
    void backpropagate(const std::string &image_path, int true_label, float learning_rate) {
        // Forward pass
        std::vector<float> vit_logits = forward(image_path);
        
        // Backpropagation: Compute gradients for classifier weights and bias
        backpropagate_vit(vit_logits, true_label, classifier_weights, classifier_bias, learning_rate);
        
        // Log classifier gradients
        std::cout << "[LOG] Gradients for classifier weights: ";
        for (const auto& row : classifier_weights_grad) {
            for (const auto& val : row) {
                std::cout << val << " ";
            }
        }
        std::cout << std::endl;
        
        std::cout << "[LOG] Gradients for classifier bias: ";
        for (const auto& grad : classifier_bias_grad) {
            std::cout << grad << " ";
        }
        std::cout << std::endl;

        // Gradient clipping to prevent exploding gradients
        const float gradient_clip_value = 5.0f;
        for (auto& row : classifier_weights_grad) {
            for (auto& grad : row) {
                grad = std::max(std::min(grad, gradient_clip_value), -gradient_clip_value);
            }
        }
        for (auto& grad : classifier_bias_grad) {
            grad = std::max(std::min(grad, gradient_clip_value), -gradient_clip_value);
        }

        // Update classifier weights using gradients
        for (int i = 0; i < classifier_weights.size(); ++i) {
            for (int j = 0; j < classifier_weights[i].size(); ++j) {
                classifier_weights[i][j] -= learning_rate * classifier_weights_grad[i][j];
            }
        }
        
        // Update classifier bias using gradients
        for (int i = 0; i < classifier_bias.size(); ++i) {
            classifier_bias[i] -= learning_rate * classifier_bias_grad[i];
        }

        // Log updated classifier weights and bias after backpropagation
        std::cout << "[LOG] Updated classifier weights: ";
        for (const auto& row : classifier_weights) {
            for (const auto& val : row) {
                std::cout << val << " ";
            }
        }
        std::cout << std::endl;
        
        std::cout << "[LOG] Updated classifier bias: ";
        for (const auto& val : classifier_bias) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
};

#endif // VIT_MODEL_HPP
