#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <stdexcept>
#include <functional>
#include <cstdlib>
#include <cmath>
#include <mutex>
#include <future>
#include <atomic>
#include <limits>

#include "vit_model.hpp"    
#include "utils.hpp"    
#include "gradients.hpp"    
#include "optimizers.hpp" 
#include "training_utils.hpp"
#include "checkpoint.hpp" 

namespace fs = std::filesystem;

std::atomic<int> processed_samples(0);  // Track number of processed images.
std::mutex update_mutex;  // Protects model parameter updates and loss accumulation.

int get_label_from_directory(const std::string &dirname) {       // Hardcoded for now
    if (dirname.find("apple") != std::string::npos) return 0;
    if (dirname.find("banana") != std::string::npos) return 1;
    if (dirname.find("orange") != std::string::npos) return 2;
    return -1;  // Unknown class
}

// Structure to hold the training data (image path and label).
struct ImageSample {
    std::string path;
    int label;
};

// Gather all image samples from the training directory.
std::vector<ImageSample> gather_training_samples(const std::string &train_dir) {
    std::vector<ImageSample> samples;
    if (!fs::exists(train_dir)) {
        throw std::runtime_error("[ERROR] Training directory does not exist: " + train_dir);
    }
    if (!fs::is_directory(train_dir)) {
        throw std::runtime_error("[ERROR] Path is not a directory: " + train_dir);
    }

    // Iterate over each entry in the training directory
    for (const auto &class_entry : fs::directory_iterator(train_dir)) {    
        if (!class_entry.is_directory()) continue;  // Check if the current entry is a directory, if not skip it
        std::string class_dir = class_entry.path().string();
        int label = get_label_from_directory(class_dir);
        if (label < 0) continue;  // If the label is invalid, skip this directory
        for (const auto &img_entry : fs::directory_iterator(class_dir)) {
            if (!img_entry.is_regular_file()) continue;
            std::string image_path = img_entry.path().string();  // Convert the path to a string and store it in image_path
            samples.push_back({image_path, label});  // Add the image path and its label as a new sample to the samples vector
        }
    }
    return samples;
}

// Batch processing: split samples into batches.
std::vector<std::vector<ImageSample>> create_batches(const std::vector<ImageSample> &samples, int batch_size) {
    if (batch_size <= 0) {
        throw std::invalid_argument("[ERROR] Batch size must be greater than zero.");
    }

    std::vector<std::vector<ImageSample>> batches;
    int total = samples.size();
    for (int i = 0; i < total; i += batch_size) {
        int end = std::min(total, i + batch_size);  // Calculate the end index of current batch, ensuring it does not exceed total number of samples
        std::vector<ImageSample> batch(samples.begin() + i, samples.begin() + end);  // Create new batch by slicing the samples vector from i to end
        batches.push_back(batch);  // Add the current batch to batches vector
    }
    return batches;
}


// Process batch with thread safety using backpropagation.
void process_batch(VitModelWrapper &model,
                   const std::vector<ImageSample> &batch,
                   float learning_rate,
                   int &global_t,
                   std::vector<std::vector<float>> &m_weights,
                   std::vector<std::vector<float>> &v_weights,
                   std::vector<float> &m_bias,
                   std::vector<float> &v_bias,
                   float &batch_loss,
                   int &batch_count) {
    std::cout << "[DEBUG] Starting batch processing with " << batch.size() << " samples." << std::endl;

    // Accumulate gradients for all samples in the batch
    for (const auto &sample : batch) {
        try {
            // Compute the forward pass and loss for the current sample
            float loss_val = model.compute_loss(sample.path, sample.label);

            if (is_nan_or_inf(loss_val)) {
                std::cerr << "[ERROR] NaN or Inf detected in loss for sample " << sample.path << std::endl;
                continue;  // Skip this sample and proceed with others.
            }

            std::cout << "[DEBUG] Loss for sample " << sample.path << ": " << loss_val << std::endl;

            {
                std::lock_guard<std::mutex> lock(update_mutex);
                batch_loss += loss_val;
                batch_count++;
            }

            // Perform the backward pass to compute gradients
            model.backpropagate(sample.path, sample.label, learning_rate);  // Backpropagation computes gradients for weights/biases

            // Debugging gradients
            for (int i = 0; i < model.embedding_dim; ++i) {
                for (int j = 0; j < model.num_classes; ++j) {
                    if (is_nan_or_inf(model.classifier_weights_grad[i][j])) {
                        std::cerr << "[ERROR] NaN or Inf detected in gradient for classifier weights at (" << i << ", " << j << ")" << std::endl;
                    }
                }
            }

            // Update model weights using Adam optimizer
            for (int i = 0; i < model.embedding_dim; ++i) {
                for (int j = 0; j < model.num_classes; ++j) {
                    // Perform Adam update for classifier weights
                    float grad = model.classifier_weights_grad[i][j];  // Get the gradient for classifier weights
                    m_weights[i][j] = 0.9f * m_weights[i][j] + (1 - 0.9f) * grad;  // Update the moving average of gradients
                    v_weights[i][j] = 0.999f * v_weights[i][j] + (1 - 0.999f) * grad * grad;  // Update the moving average of gradients but *squared*
                    float m_hat = m_weights[i][j] / (1 - std::pow(0.9f, global_t));  // Compute biased first moment estimate.
                    float v_hat = v_weights[i][j] / (1 - std::pow(0.999f, global_t));  // Compute biased second moment estimate.

                    if (is_nan_or_inf(m_hat) || is_nan_or_inf(v_hat)) {
                        std::cerr << "[ERROR] NaN or Inf detected in Adam update for classifier weights at (" << i << ", " << j << ")" << std::endl;
                    }

                    model.classifier_weights[i][j] -= learning_rate * m_hat / (std::sqrt(v_hat) + 1e-8f);  // Update weights
                }
            }

            // Perform Adam update for classifier biases
            // Same as above but bias
            for (int j = 0; j < model.num_classes; ++j) {
                float grad = model.classifier_bias_grad[j];
                m_bias[j] = 0.9f * m_bias[j] + (1 - 0.9f) * grad;
                v_bias[j] = 0.999f * v_bias[j] + (1 - 0.999f) * grad * grad;
                float m_hat = m_bias[j] / (1 - std::pow(0.9f, global_t));
                float v_hat = v_bias[j] / (1 - std::pow(0.999f, global_t));

                if (is_nan_or_inf(m_hat) || is_nan_or_inf(v_hat)) {
                    std::cerr << "[ERROR] NaN or Inf detected in Adam update for classifier bias at (" << j << ")" << std::endl;
                }

                model.classifier_bias[j] -= learning_rate * m_hat / (std::sqrt(v_hat) + 1e-8f);
            }

            {
                std::lock_guard<std::mutex> lock(update_mutex);  // Multi threading moment
                global_t++;  // Increment global time step.
            }

        } catch (const std::exception &e) {
            std::lock_guard<std::mutex> lock(update_mutex);
            std::cerr << "[DEBUG] Error processing " << sample.path << ": " << e.what() << std::endl;
        }
    }

    std::cout << "[DEBUG] Batch processed with total batch loss: " << batch_loss << " and total batch count: " << batch_count << std::endl;
}

void train_model(const std::string &train_dir, int num_classes, int num_blocks, int epochs, float learning_rate, int batch_size, int max_workers) {
    std::string checkpoint_file = "model_checkpoint.bin";

    // Load checkpoint if it exists
    VitModelWrapper model(768, num_classes, num_blocks);
    int epoch = 0, global_t = 1;
    std::vector<std::vector<float>> m_weights, v_weights;
    std::vector<float> m_bias, v_bias;

    if (!load_checkpoint(checkpoint_file, model, epoch, global_t, m_weights, v_weights, m_bias, v_bias)) {
        std::cout << "[DEBUG] No checkpoint found, starting from scratch." << std::endl;

        // Initialize Adam optimizer variables if starting from scratch
        m_weights.resize(768, std::vector<float>(num_classes, 0.0f));  // Adjust dimensions: embedding_dim (768) and num_classes
        v_weights.resize(768, std::vector<float>(num_classes, 0.0f));
        m_bias.resize(num_classes, 0.0f);
        v_bias.resize(num_classes, 0.0f);
    }

    // Gather all training samples
    std::vector<ImageSample> samples = gather_training_samples(train_dir);
    if (samples.empty()) {
        throw std::runtime_error("[ERROR] No training samples found!");
    }
    std::cout << "[DEBUG] Total training samples: " << samples.size() << std::endl;

    // Create batches
    auto batches = create_batches(samples, batch_size);
    std::cout << "[DEBUG] Created " << batches.size() << " batches (batch size = " << batch_size << ")." << std::endl;

    EarlyStopping early_stopping(3);

    // Training loop over epochs
    for (epoch = epoch; epoch < epochs; ++epoch) {
        float epoch_loss = 0.0f;
        int sample_count = 0;
        std::cout << "[DEBUG] Epoch " << (epoch + 1) << "/" << epochs << std::endl;

        std::vector<std::future<void>> futures;

        for (auto &batch : batches) {
            if (futures.size() >= max_workers) {  // If no of workers exceed max workers
                for (auto &fut : futures) {
                    fut.get();  // Wait for task to finish
                }
                futures.clear();
            }

            // Create a new task for a batch
            futures.push_back(std::async(std::launch::async, [&]() {
                float batch_loss = 0.0f;
                int batch_count = 0;
                std::cout << "[DEBUG] Processing batch with " << batch_count << " samples." << std::endl;
                process_batch(model, batch, learning_rate, global_t, m_weights, v_weights, m_bias, v_bias, batch_loss, batch_count);

                {
                    std::lock_guard<std::mutex> lock(update_mutex);  // Lock mutex to update shared variables
                    epoch_loss += batch_loss;
                    sample_count += batch_count;
                }
            }));
        }

        for (auto &fut : futures) {
            fut.get();  // Wait fr all tasks are complete
        }

        if (sample_count > 0) {
            float avg_loss = epoch_loss / sample_count;
            std::cout << "[DEBUG] Epoch " << (epoch + 1) << " average loss: " << avg_loss << std::endl;
            early_stopping.update(avg_loss);
            if (early_stopping.stop) {
                std::cout << "[DEBUG] Early stopping triggered. Stopping training." << std::endl;
                break;
            }
        } else {
            std::cout << "[DEBUG] No training samples processed in epoch " << (epoch + 1) << std::endl;
        }

        // Save checkpoint after each epoch
        save_checkpoint(checkpoint_file, model, epoch, global_t, m_weights, v_weights, m_bias, v_bias);
    }

    std::cout << "\n[DEBUG] Training completed." << std::endl;
}
