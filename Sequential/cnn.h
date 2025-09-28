#include <fstream>
#include <cstring>
#include "layers.h"

// CNN Structure
class CNN {
public:
    // Convolutional Layer 1 (C1): 28x28 -> 6x24x24
    float c1_weight[6][5][5];
    float c1_bias[6];
    float c1_preact[6][24][24];
    float c1_output[6][24][24];
    float d_c1_weight[6][5][5];
    float d_c1_preact[6][24][24];
    float d_c1_output[6][24][24];

    // Subsampling Layer 1 (S1): 6x24x24 -> 6x6x6
    float s1_weight[1][4][4];
    float s1_bias[1];
    float s1_preact[6][6][6];
    float s1_output[6][6][6];
    float d_s1_weight[1][4][4];
    float d_s1_preact[6][6][6];
    float d_s1_output[6][6][6];

    // Fully Connected Layer (F): 216 (6x6x6 flattened) -> 10
    float f_weight[10][216];
    float f_bias[10];
    float f_preact[10];
    float f_output[10];
    float d_f_weight[10][216];
    float d_f_preact[10];

    CNN() {
        initializeWeights();
        initializeGradients();
    }

    // to load a saved model for testing purposes.
    bool loadModel(const std::string& filename) {
        std::ifstream model_file(filename, std::ios::binary);
        if (!model_file.is_open()) {
            return false;
        }
        
        // Load C1 weights and biases
        model_file.read(reinterpret_cast<char*>(c1_weight), sizeof(c1_weight));
        model_file.read(reinterpret_cast<char*>(c1_bias), sizeof(c1_bias));
        
        // Load S1 weights and biases
        model_file.read(reinterpret_cast<char*>(s1_weight), sizeof(s1_weight));
        model_file.read(reinterpret_cast<char*>(s1_bias), sizeof(s1_bias));
        
        // Load F weights and biases
        model_file.read(reinterpret_cast<char*>(f_weight), sizeof(f_weight));
        model_file.read(reinterpret_cast<char*>(f_bias), sizeof(f_bias));
        
        model_file.close();
        return true;
    }

    void initializeWeights() {
        // Use the initialization functions from layers.h
        initializeC1Weights(c1_weight, c1_bias);
        initializeS1Weights(s1_weight, s1_bias);
        initializeFWeights(f_weight, f_bias);
    }

    void initializeGradients() {
        // Initialize all gradients to zero
        memset(d_c1_weight, 0, sizeof(d_c1_weight));
        memset(d_c1_preact, 0, sizeof(d_c1_preact));
        memset(d_c1_output, 0, sizeof(d_c1_output));
        memset(d_s1_weight, 0, sizeof(d_s1_weight));
        memset(d_s1_preact, 0, sizeof(d_s1_preact));
        memset(d_s1_output, 0, sizeof(d_s1_output));
        memset(d_f_weight, 0, sizeof(d_f_weight));
        memset(d_f_preact, 0, sizeof(d_f_preact));
    }

    void forward(const float input[28][28]) {
        // Forward pass through C1
        fp_c1(input, c1_preact, c1_weight, c1_bias);
        applyActivationC1(c1_preact, c1_output);

        // Forward pass through S1
        fp_s1(c1_output, s1_preact, s1_weight, s1_bias);
        applyActivationS1(s1_preact, s1_output);

        // Forward pass through F
        fp_preact_f(s1_output, f_preact, f_weight);
        fp_bias_f(f_preact, f_bias);
        applyActivationF(f_preact, f_output);
    }

    void backward(int target_label) {
        // For softmax + cross-entropy, the gradient is simply: softmax_output - target
        // Use the function from layers.h
        softmax_cross_entropy_gradient(f_output, target_label, d_f_preact);

        // Backprop through F layer
        bp_weight_f(d_f_weight, d_f_preact, s1_output);
        bp_bias_f(f_bias, d_f_preact);
        bp_output_s1(d_s1_output, f_weight, d_f_preact);

        // Backprop through S1 layer
        bp_preact_s1(d_s1_preact, d_s1_output, s1_preact);
        bp_weight_s1(d_s1_weight, d_s1_preact, c1_output);
        bp_bias_s1(s1_bias, d_s1_preact);
        bp_output_c1(d_c1_output, s1_weight, d_s1_preact);

        // Backprop through C1 layer
        bp_preact_c1(d_c1_preact, d_c1_output, c1_preact);
        // Note: bp_weight_c1 will be called in trainCNN function with actual input
        bp_bias_c1(c1_bias, d_c1_preact);
    }

    void resetGradients() {
        // Reset gradients before each training iteration
        memset(d_c1_weight, 0, sizeof(d_c1_weight));
        memset(d_c1_preact, 0, sizeof(d_c1_preact));
        memset(d_c1_output, 0, sizeof(d_c1_output));
        memset(d_s1_weight, 0, sizeof(d_s1_weight));
        memset(d_s1_preact, 0, sizeof(d_s1_preact));
        memset(d_s1_output, 0, sizeof(d_s1_output));
        memset(d_f_weight, 0, sizeof(d_f_weight));
        memset(d_f_preact, 0, sizeof(d_f_preact));
    }

    void updateWeights() {
        // Use the update functions from layers.h
        updateFWeights(f_weight, d_f_weight);
        updateS1Weights(s1_weight, d_s1_weight);
        updateC1Weights(c1_weight, d_c1_weight);
    }

    int predict(const float input[28][28]) {
        forward(input);
        int max_idx = 0;
        float max_val = f_output[0];
        for (int i = 1; i < 10; ++i) {
            if (f_output[i] > max_val) {
                max_val = f_output[i];
                max_idx = i;
            }
        }
        return max_idx;
    }

    // Cross entropy loss for softmax output
    float calculateLoss(int target_label) {
        // For softmax + cross-entropy: loss = -log(p_correct_class)
        float pred_clamped = std::max(epsilon, f_output[target_label]);
        return -std::log(pred_clamped);
    }
};