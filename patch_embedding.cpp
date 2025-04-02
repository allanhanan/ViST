#include <iostream>
#include <vector>
#include <cmath>

// Split the image into patches and embed them
std::vector<std::vector<float>> create_patch_embedding(const std::vector<std::vector<std::vector<float>>>& image_matrix, int patch_size) {
    // Get image dimensions and channels
    int height = image_matrix.size();
    int width = image_matrix[0].size();
    int channels = image_matrix[0][0].size();
    
    // Calculate number of patches along each dimension (rounding up to cover all pixels)
    int num_patches_y = std::ceil((float)height / patch_size);
    int num_patches_x = std::ceil((float)width / patch_size);
    
    std::vector<std::vector<float>> patch_embeddings(num_patches_y * num_patches_x, std::vector<float>(patch_size * patch_size * channels));
    
    // Iterate through each patch in the image
    for (int patch_y = 0; patch_y < num_patches_y; ++patch_y) {
        for (int patch_x = 0; patch_x < num_patches_x; ++patch_x) {
            int patch_index = patch_y * num_patches_x + patch_x; // Calculate patch index
            
            // Iterate through each pixel in the current patch
            for (int y = patch_y * patch_size; y < std::min((patch_y + 1) * patch_size, height); ++y) {
                for (int x = patch_x * patch_size; x < std::min((patch_x + 1) * patch_size, width); ++x) {
                    // Calculate pixel offset within the patch
                    int patch_offset = ((y - patch_y * patch_size) * patch_size + (x - patch_x * patch_size)) * channels;
                    
                    // Iterate through each channel and store pixel value in embedding
                    for (int c = 0; c < channels; ++c) {
                        patch_embeddings[patch_index][patch_offset + c] = static_cast<float>(image_matrix[y][x][c]);
                    }
                }
            }
        }
    }
    
    return patch_embeddings;
}