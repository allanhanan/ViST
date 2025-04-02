#ifndef IMAGE_UTILS_HPP
#define IMAGE_UTILS_HPP

#include <vector>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <random>

// Function to apply a random horizontal flip to the image
void random_flip(std::vector<float>& image, int width, int height) {
    if (rand() % 2) {  // Randomly decide whether to flip or not
        for (int row = 0; row < height; ++row) {  // Iterate through each row
            for (int col = 0; col < width / 2; ++col) {  // Only iterate up to half the columns
                int idx1 = (row * width + col) * 3;  // Calculate index of first pixel in the pair
                int idx2 = (row * width + (width - col - 1)) * 3;  // Calculate index of second pixel in the pair
                std::swap(image[idx1], image[idx2]);  // Swap the RGB values of the two pixels
                std::swap(image[idx1 + 1], image[idx2 + 1]);
                std::swap(image[idx1 + 2], image[idx2 + 2]);
            }
        }
    }
}

// Function to rotate the image by 90 degrees clockwise
void rotate_90(std::vector<float>& image, int& width, int& height) {
    std::vector<float> rotated_image(image.size());  // Create a new vector to store the rotated image
    int new_width = height;  // New width after rotation
    int new_height = width;  // New height after rotation
    
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            int src_idx = (row * width + col) * 3;  // Calculate source index
            int dst_idx = ((col * new_width) + (new_height - row - 1)) * 3;  // Calculate destination index
            rotated_image[dst_idx] = image[src_idx];
            rotated_image[dst_idx + 1] = image[src_idx + 1];
            rotated_image[dst_idx + 2] = image[src_idx + 2];
        }
    }
    image = std::move(rotated_image);  // Move the rotated image back to the original vector
    width = new_width;
    height = new_height;
}

// Function to rotate an image 180 degrees
void rotate_180(std::vector<float>& image, int width, int height) {
    for (int row = 0; row < height / 2; ++row) {
        for (int col = 0; col < width; ++col) {
            int idx1 = (row * width + col) * 3;
            int idx2 = ((height - row - 1) * width + (width - col - 1)) * 3;
            std::swap(image[idx1], image[idx2]);
            std::swap(image[idx1 + 1], image[idx2 + 1]);
            std::swap(image[idx1 + 2], image[idx2 + 2]);
        }
    }
}

// Function to rotate an image 270 degrees
void rotate_270(std::vector<float>& image, int& width, int& height) {
    rotate_90(image, width, height);
    rotate_90(image, width, height);
    rotate_90(image, width, height);
}

// Function to adjust the brightness of the image
void random_brightness(std::vector<float>& image) {
    float factor = 0.5f + (rand() % 5) * 0.1f;  // Random factor between 0.5 and 1.5
    for (auto& pixel : image) {
        pixel = std::min(std::max(pixel * factor, 0.0f), 1.0f);  // Clamp between [0, 1]
    }
}

// Function to apply random brightness and contrast adjustment
void random_brightness_contrast(std::vector<float>& image) {
    random_brightness(image);  // Can be extended to add contrast as well
}

// Function to apply data augmentation (flip, rotate, brightness)
void apply_data_augmentation(std::vector<float>& image, int width, int height) {
    random_flip(image, width, height);
    
    int rotate_angle = rand() % 4;  // Randomly choose 0, 90, 180, or 270 degrees
    switch (rotate_angle) {
        case 0: break;   // No rotation
        case 1: rotate_90(image, width, height); break;
        case 2: rotate_180(image, width, height); break;
        case 3: rotate_270(image, width, height); break;
    }

    random_brightness_contrast(image);  // Adjust brightness/contrast
}

#endif // IMAGE_UTILS_HPP
