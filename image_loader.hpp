#ifndef IMAGE_LOADER_HPP
#define IMAGE_LOADER_HPP

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <iostream>
#include <vector>
#include <string>

// ResizeS image using nearest-neighbor interpolation
std::vector<std::vector<std::vector<float>>> resize_image(
    const std::vector<std::vector<std::vector<float>>>& original_image, 
    int original_width, int original_height, int channels, 
    int target_width, int target_height) 
{
    // Initialize the resized image with dimensions (target_height, target_width, 3)
    std::vector<std::vector<std::vector<float>>> resized_image(target_height, 
        std::vector<std::vector<float>>(target_width, std::vector<float>(3))); 

    for (int y = 0; y < target_height; ++y) {
        for (int x = 0; x < target_width; ++x) {
            // Calculate the corresponding pixel in the original image
            int orig_x = (x * original_width) / target_width;
            int orig_y = (y * original_height) / target_height;

            // Copy the pixel values from the original image to the resized image
            for (int c = 0; c < 3; ++c) {     // Only copy 3 channels (RGB)
                resized_image[y][x][c] = original_image[orig_y][orig_x][c];
            }
        }
    }

    return resized_image;
}

// Load image from a file
std::vector<std::vector<std::vector<float>>> load_image(const std::string& filename, int target_width, int target_height, int& width, int& height, int& channels) {
    // Load the image data using stb_load and get the dimensions
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, &channels, 0);

    if (data) {
        std::cout << "Image loaded successfully!" << std::endl;
        std::cout << "Width: " << width << ", Height: " << height << ", Channels: " << channels << std::endl;

        // Initialize the image matrix with dimensions (height, width, 3)
        std::vector<std::vector<std::vector<float>>> image_matrix(height, 
            std::vector<std::vector<float>>(width, std::vector<float>(3))); 

        // Fill the image_matrix with loaded pixel values
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = (y * width + x) * channels;
                // Only use the first 3 channels (RGB)
                image_matrix[y][x][0] = static_cast<float>(data[idx]) / 255.0f;       // R
                image_matrix[y][x][1] = static_cast<float>(data[idx + 1]) / 255.0f;   // G
                image_matrix[y][x][2] = static_cast<float>(data[idx + 2]) / 255.0f;   // B
            }
        }

        stbi_image_free(data);

        // Resize the image if necessary
        if (width != target_width || height != target_height) {
            std::cout << "[LOG] Resizing image to: " << target_width << "x" << target_height << std::endl;
            return resize_image(image_matrix, width, height, channels, target_width, target_height);
        }

        return image_matrix;
    } else {
        std::cerr << "Failed to load image!" << std::endl;
        return {};
    }
}

#endif // IMAGE_LOADER_HPP