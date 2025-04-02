
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include "vit_model.hpp" 
#include "checkpoint.hpp"    
#include "image_loader.hpp"   

// Define class labels for the image classification task
const std::vector<std::string> class_labels = {"apple", "banana", "orange"};


void test_model(const std::string& checkpoint_file, const std::string& image_path, VitModelWrapper& model) {
    // Initialize Adam optimizer variables (required for checkpoint loading)
    std::vector<std::vector<float>> m_weights, v_weights;
    std::vector<float> m_bias, v_bias;
    int epoch = 0, global_t = 0;
    
    // Load the model from the checkpoint
    if (!load_checkpoint(checkpoint_file, model, epoch, global_t, m_weights, v_weights, m_bias, v_bias)) {
        std::cerr << "[ERROR] Could not load checkpoint from " << checkpoint_file << std::endl;
        return;
    }
    std::cout << "[LOG] Loaded model from checkpoint. Ready for inference." << std::endl;


    int target_width = 300;
    int target_height = 300;
    int width, height, channels;
    

    std::cout << "[LOG] Loading and preprocessing image: " << image_path << std::endl;
    auto image_matrix = load_image(image_path, target_width, target_height, width, height, channels);
    
    if (image_matrix.empty()) {
        std::cerr << "[ERROR] Failed to load or preprocess image: " << image_path << std::endl;
        return;
    }

    // Apply patch embedding
    int patch_size = 16;  // Same patch size as used in training
    
    // Log the action of creating patch embeddings
    std::cout << "[LOG] Creating patch embeddings..." << std::endl;
    auto patch_embeddings = create_patch_embedding(image_matrix, patch_size);
    
    if (patch_embeddings.empty()) {
        std::cerr << "[ERROR] Failed to create patch embeddings!" << std::endl;
        return;
    }

    std::cout << "[LOG] Number of patches: " << patch_embeddings.size()
              << ", Patch embedding size: " << patch_embeddings[0].size() << std::endl;

    // Apply positional encoding
    int num_patches = patch_embeddings.size();
    
    std::cout << "[LOG] Creating positional encoding..." << std::endl;
    auto pos_encoding = create_positional_encoding(num_patches, model.embedding_dim);
    auto tokens_with_pos = add_positional_encoding(patch_embeddings, pos_encoding);

    // Add the [CLS] token (learnable token)
    std::cout << "[LOG] Adding [CLS] token..." << std::endl;
    std::vector<float> cls_token = random_vector(model.embedding_dim);  // Learnable token
    std::vector<std::vector<float>> tokens;
    tokens.push_back(cls_token);  // Add [CLS] token
    
    // Insert the positional encoded patches after the [CLS] token
    tokens.insert(tokens.end(), tokens_with_pos.begin(), tokens_with_pos.end());

    // Perform forward pass to get logits
    std::cout << "[LOG] Running inference through Vision Transformer..." << std::endl;
    
    std::vector<float> logits = model.forward(image_path);
    
    if (logits.empty()) {
        std::cerr << "[ERROR] Model returned empty logits. Inference failed." << std::endl;
        return;
    }

    std::cout << "[RESULT] Logits for all classes:" << std::endl;
    
    for (size_t i = 0; i < logits.size(); ++i) {
        std::cout << class_labels[i] << ": " << logits[i] << std::endl;
    }

    // Determine the predicted class based on the highest logit
    int predicted_class = std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
    
    std::cout << "[RESULT] Predicted Class: " << class_labels[predicted_class] << std::endl;
}
