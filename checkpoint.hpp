#ifndef CHECKPOINT_HPP
#define CHECKPOINT_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include "vit_model.hpp"  // Include VitModelWrapper definition

// Save model and optimizer state to a checkpoint file
bool save_checkpoint(const std::string& filename,
                     const VitModelWrapper& model,
                     int epoch,
                     int global_t,
                     const std::vector<std::vector<float>>& m_weights,
                     const std::vector<std::vector<float>>& v_weights,
                     const std::vector<float>& m_bias,
                     const std::vector<float>& v_bias) {
    // Open file for binary writing
    std::ofstream out_file(filename, std::ios::binary);
    if (!out_file) {
        std::cerr << "Error: Could not open checkpoint file for writing!" << std::endl;
        return false;
    }

    // Save model parameters
    size_t embedding_dim = model.embedding_dim;  // Get the dimension of the model's embeddings
    size_t num_classes = model.num_classes;      // Get the number of classes in the model

    out_file.write(reinterpret_cast<const char*>(&embedding_dim), sizeof(embedding_dim));  // Write embedding dimension to file
    out_file.write(reinterpret_cast<const char*>(&num_classes), sizeof(num_classes));      // Write number of classes to file

    // Save classifier weights and biases
    for (size_t i = 0; i < embedding_dim; ++i) {
        out_file.write(reinterpret_cast<const char*>(&model.classifier_weights[i][0]), num_classes * sizeof(float));  // Write each weight
    }
    out_file.write(reinterpret_cast<const char*>(&model.classifier_bias[0]), num_classes * sizeof(float));             // Write all biases

    // Save ADAM optimizer states (momentum and velocity weights, momentum and velocity biases)
    for (size_t i = 0; i < embedding_dim; ++i) {
        out_file.write(reinterpret_cast<const char*>(&m_weights[i][0]), num_classes * sizeof(float));  // Write each momentum weight
        out_file.write(reinterpret_cast<const char*>(&v_weights[i][0]), num_classes * sizeof(float));  // Write each velocity weight
    }
    out_file.write(reinterpret_cast<const char*>(&m_bias[0]), num_classes * sizeof(float));              // Write all momentum biases
    out_file.write(reinterpret_cast<const char*>(&v_bias[0]), num_classes * sizeof(float));               // Write all velocity biases

    // Save training progress
    out_file.write(reinterpret_cast<const char*>(&epoch), sizeof(epoch));  // Write the current epoch to file
    out_file.write(reinterpret_cast<const char*>(&global_t), sizeof(global_t));  // Write the global step counter to file


    out_file.close();
    std::cout << "[LOG] Checkpoint saved successfully to " << filename << std::endl;
    return true;
}

// Load model and optimizer state from a checkpoint file
bool load_checkpoint(const std::string& filename,
                     VitModelWrapper& model,
                     int& epoch,
                     int& global_t,
                     std::vector<std::vector<float>>& m_weights,
                     std::vector<std::vector<float>>& v_weights,
                     std::vector<float>& m_bias,
                     std::vector<float>& v_bias) {
    // Open file for binary reading
    std::ifstream in_file(filename, std::ios::binary);
    if (!in_file) {
        std::cerr << "Error: Could not open checkpoint file for reading!" << std::endl;
        return false;
    }

    size_t embedding_dim;  // Variable to store the dimension of the model's embeddings
    size_t num_classes;    // Variable to store the number of classes in the model

    // Load model parameters
    in_file.read(reinterpret_cast<char*>(&embedding_dim), sizeof(embedding_dim));  // Read embedding dimension from file
    in_file.read(reinterpret_cast<char*>(&num_classes), sizeof(num_classes));      // Read number of classes from file

    // Ensure model dimensions match checkpoint
    if (model.embedding_dim != embedding_dim || model.num_classes != num_classes) {
        std::cerr << "Error: Model dimensions do not match checkpoint!" << std::endl;
        return false;
    }

    // Pre-allocate memory for optimizer states (m_weights, v_weights, m_bias, v_bias)
    m_weights.resize(embedding_dim, std::vector<float>(num_classes, 0.0f));  // Pre-allocate momentum weights
    v_weights.resize(embedding_dim, std::vector<float>(num_classes, 0.0f));  // Pre-allocate velocity weights
    m_bias.resize(num_classes, 0.0f);                                    // Pre-allocate momentum biases
    v_bias.resize(num_classes, 0.0f);                                     // Pre-allocate velocity biases

    // Load classifier weights and biases
    for (size_t i = 0; i < embedding_dim; ++i) {
        in_file.read(reinterpret_cast<char*>(&model.classifier_weights[i][0]), num_classes * sizeof(float));  // Read each weight
    }
    in_file.read(reinterpret_cast<char*>(&model.classifier_bias[0]), num_classes * sizeof(float));             // Read all biases

    // Load optimizer states (momentum and velocity weights, momentum and velocity biases)
    for (size_t i = 0; i < embedding_dim; ++i) {
        in_file.read(reinterpret_cast<char*>(&m_weights[i][0]), num_classes * sizeof(float));  // Read each momentum weight
        in_file.read(reinterpret_cast<char*>(&v_weights[i][0]), num_classes * sizeof(float));  // Read each velocity weight
    }
    in_file.read(reinterpret_cast<char*>(&m_bias[0]), num_classes * sizeof(float));              // Read all momentum biases
    in_file.read(reinterpret_cast<char*>(&v_bias[0]), num_classes * sizeof(float));               // Read all velocity biases

    // Load training progress
    in_file.read(reinterpret_cast<char*>(&epoch), sizeof(epoch));  // Read the current epoch from file
    in_file.read(reinterpret_cast<char*>(&global_t), sizeof(global_t));  // Read the global step counter from file


    in_file.close();
    std::cout << "[LOG] Checkpoint loaded successfully from " << filename << std::endl;
    return true;
}

#endif // CHECKPOINT_HPP