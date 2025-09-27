#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include "layers.h"
#include "dataloader.h"
#include "cnn.h"

// Clean training function using CNN's built-in methods
void trainCNN(CNN& cnn, const float input[28][28], int target_label) {
    // Reset gradients before each training iteration
    cnn.resetGradients();
    
    // Forward pass
    cnn.forward(input);
    
    // Backward pass using CNN's method
    cnn.backward(target_label);
    
    // Note: We still need to call bp_weight_c1 separately as it needs the input
    bp_weight_c1(cnn.d_c1_weight, cnn.d_c1_preact, input);
    
    // Update weights
    cnn.updateWeights();
}

int main() {
    std::cout << "Loading MNIST dataset..." << std::endl;
    
    // Load training data
    MNISTData train_data = loadMNISTImages("../train-images-idx3-ubyte");
    std::vector<int> train_labels = loadMNISTLabels("../train-labels-idx1-ubyte");

    // Load test data
    MNISTData test_data = loadMNISTImages("../t10k-images-idx3-ubyte");
    std::vector<int> test_labels = loadMNISTLabels("../t10k-labels-idx1-ubyte");

    if (train_data.num_images == 0 || train_labels.empty()) {
        std::cerr << "Failed to load MNIST dataset. Please ensure the following files are in the current directory:" << std::endl;
        return -1;
    }

    std::cout << "Loaded " << train_data.num_images << " training images" << std::endl;
    std::cout << "Loaded " << test_data.num_images << " test images" << std::endl;

    // Initialize CNN
    CNN cnn;
    
    // Training parameters
    const int epochs = 5;
    const int batch_size = 100;
    const int num_batches = train_data.num_images / batch_size;

    std::cout << "Starting training..." << std::endl;
    std::cout << "Epochs: " << epochs << ", Batch size: " << batch_size << std::endl;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;
        int correct_predictions = 0;

        // Shuffle training data
        std::vector<int> indices(train_data.num_images);
        std::iota(indices.begin(), indices.end(), 0);
        std::random_shuffle(indices.begin(), indices.end());

        for (int batch = 0; batch < num_batches; ++batch) {
            float batch_loss = 0.0f;
            
            for (int i = 0; i < batch_size; ++i) {
                int idx = indices[batch * batch_size + i];
                
                // Convert 1D image to 28x28 array
                float input[28][28];
                for (int r = 0; r < 28; ++r) {
                    for (int c = 0; c < 28; ++c) {
                        input[r][c] = train_data.images[idx][r * 28 + c];
                    }
                }

                // Train the CNN
                trainCNN(cnn, input, train_labels[idx]);
                
                // Calculate loss and accuracy
                float loss = cnn.calculateLoss(train_labels[idx]);
                batch_loss += loss;
                
                int prediction = cnn.predict(input);
                if (prediction == train_labels[idx]) {
                    correct_predictions++;
                }
            }
            
            total_loss += batch_loss / batch_size;
            
            // Print progress every 100 batches
            if ((batch + 1) % 100 == 0) {
                std::cout << "Epoch " << (epoch + 1) << "/" << epochs 
                          << ", Batch " << (batch + 1) << "/" << num_batches 
                          << ", Loss: " << std::fixed << std::setprecision(4) << (batch_loss / batch_size) << std::endl;
            }
        }

        // Calculate epoch accuracy
        float accuracy = static_cast<float>(correct_predictions) / (num_batches * batch_size) * 100.0f;
        std::cout << "Epoch " << (epoch + 1) << " completed. Average Loss: " << std::fixed << std::setprecision(4) 
                  << (total_loss / num_batches) << ", Training Accuracy: " << std::setprecision(2) << accuracy << "%" << std::endl;

        // Test on validation set every epoch
        if (test_data.num_images > 0) {
            int test_correct = 0;
            int test_samples = std::min(1000, test_data.num_images); // Test on first 1000 samples for speed
            
            for (int i = 0; i < test_samples; ++i) {
                float input[28][28];
                for (int r = 0; r < 28; ++r) {
                    for (int c = 0; c < 28; ++c) {
                        input[r][c] = test_data.images[i][r * 28 + c];
                    }
                }
                
                int prediction = cnn.predict(input);
                if (prediction == test_labels[i]) {
                    test_correct++;
                }
            }
            
            float test_accuracy = static_cast<float>(test_correct) / test_samples * 100.0f;
            std::cout << "Test Accuracy: " << std::setprecision(2) << test_accuracy << "%" << std::endl;
        }
        
        std::cout << std::endl;
    }

    std::cout << "Training completed!" << std::endl;
    
    // Save the trained model
    std::cout << "Saving trained model..." << std::endl;
    
    std::ofstream model_file("smolCNN.bin", std::ios::binary);
    if (model_file.is_open()) {
        // Save C1 weights and biases
        model_file.write(reinterpret_cast<char*>(cnn.c1_weight), sizeof(cnn.c1_weight));
        model_file.write(reinterpret_cast<char*>(cnn.c1_bias), sizeof(cnn.c1_bias));
        
        // Save S1 weights and biases
        model_file.write(reinterpret_cast<char*>(cnn.s1_weight), sizeof(cnn.s1_weight));
        model_file.write(reinterpret_cast<char*>(cnn.s1_bias), sizeof(cnn.s1_bias));
        
        // Save F weights and biases
        model_file.write(reinterpret_cast<char*>(cnn.f_weight), sizeof(cnn.f_weight));
        model_file.write(reinterpret_cast<char*>(cnn.f_bias), sizeof(cnn.f_bias));
        
        model_file.close();
        std::cout << "Model saved to 'trained_model.bin'" << std::endl;
    } else {
        std::cerr << "Error: Could not save model to file" << std::endl;
    }

    return 0;
}
