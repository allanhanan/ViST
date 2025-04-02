#include <iostream>
#include <vector>
#include <cmath>

// Generate positional encodings for patches
std::vector<std::vector<float>> create_positional_encoding(int num_patches, int embedding_dim) {

    std::vector<std::vector<float>> positional_encoding(num_patches, std::vector<float>(embedding_dim));
    
    // Iterate over each patch
    for (int i = 0; i < num_patches; ++i) {
        // Iterate over each dimension in the embedding space
        for (int j = 0; j < embedding_dim; ++j) {
            float angle = (float)i / std::pow(10000, (2.0 * ((float)j / 2.0)) / (float)embedding_dim);
            // Apply sine or cosine function based on dimension index
            if (j % 2 == 0) {
                positional_encoding[i][j] = std::sin(angle);
            } else {
                positional_encoding[i][j] = std::cos(angle);
            }
        }
    }
    return positional_encoding;
}

// Add positional encoding to patch embeddings with scaling
std::vector<std::vector<float>> add_positional_encoding(
    const std::vector<std::vector<float>>& patch_embeddings, 
    const std::vector<std::vector<float>>& positional_encoding) 
{
    // Get number of patches and embedding dimension
    int num_patches = patch_embeddings.size();
    int embedding_dim = patch_embeddings[0].size();
    

    std::vector<std::vector<float>> patch_with_positional_encoding(num_patches, std::vector<float>(embedding_dim));
    
    // Scaling factor to prevent excessive influence of positional encoding
    float lambda = 0.1f; 
    
    // Iterate over each patch
    for (int i = 0; i < num_patches; ++i) {
        // Iterate over each dimension in the embedding space
        for (int j = 0; j < embedding_dim; ++j) {
            // Add positional encoding to patch embeddings with scaling
            patch_with_positional_encoding[i][j] = patch_embeddings[i][j] + lambda * positional_encoding[i][j];
        }
    }
    return patch_with_positional_encoding;
}