#include <iostream>
#include <string>
#include "train.cpp"
#include "test.cpp"

int main(int argc, char** argv) {
    if (argc == 1) {
        // Default to training mode if no arguments are passed
        std::cout << "[INFO] Training mode activated." << std::endl;

        std::string train_dir = "../train";  // Path to the training dataset
        int num_classes = 3;                // Number of classes (apple, banana, orange)
        int num_blocks = 1;                // Number of transformer blocks
        int epochs = 1;                    // Number of epochs to train
        float learning_rate = 0.001f;       // Learning rate
        int batch_size = 5;                 // Number of images in a batch
        int max_workers = 25;                // Maximum number of concurrent workers

        // Call the training function
        train_model(train_dir, num_classes, num_blocks, epochs, learning_rate, batch_size, max_workers);
    } 
    else if (argc == 3) {
        // Test mode
        std::cout << "[INFO] Test mode activated." << std::endl;

        std::string checkpoint_file = argv[1];  // Path to the checkpoint file
        std::string image_path = argv[2];       // Path to the test image

        // Call the testing function
        // Pass the checkpoint file and image path to the test function
        int embedding_dim = 768;   // Embedding dimension
        int num_classes = 3;       // Number of classes (apple, banana, orange)
        int num_blocks = 12;       // Number of transformer blocks

        VitModelWrapper model(embedding_dim, num_classes, num_blocks); // Create model instance
        test_model(checkpoint_file, image_path, model);  // Perform inference
    } 
    else {
        std::cerr << "[ERROR] Invalid arguments!" << std::endl;
        std::cerr << "[USAGE] For training: " << argv[0] << std::endl;
        std::cerr << "[USAGE] For testing: " << argv[0] << " <checkpoint_file> <image_path>" << std::endl;
        return 1;
    }

    return 0;
}
