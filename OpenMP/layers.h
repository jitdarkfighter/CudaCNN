#include <vector>
#include <memory>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>
#include <omp.h>

#ifndef LAYER_H
#define LAYER_H

const static float lr = 1.0E-02f;  // Reduced learning rate for ReLU networks
const float epsilon = 1e-15f; 

// ReLU activation function
float relu_function(float x) {
    return (x > 0.0f) ? x : 0.0f;
}

// ReLU derivative
float relu_derivative(float x) {
    return (x > 0.0f) ? 1.0f : 0.0f;
}

// Softmax activation for output layer
void softmax_function(float input[10], float output[10]) {
    float max_val = input[0];
    // Find maximum for numerical stability
    for (int i = 1; i < 10; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }
    
    // Compute exponentials and sum
    float sum = 0.0f;
    for (int i = 0; i < 10; i++) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }
    
    // Normalize
    for (int i = 0; i < 10; i++) {
        output[i] /= sum;
    }
}

// Compute gradient for softmax + cross-entropy loss
// softmax_output: output from softmax function
// target_label: true class label (0-9)
// gradient: output gradient array
void softmax_cross_entropy_gradient(float softmax_output[10], int target_label, float gradient[10]) {
    for (int i = 0; i < 10; i++) {
        if (i == target_label) {
            gradient[i] = softmax_output[i] - 1.0f;  // derivative: p_i - 1 for correct class
        } else {
            gradient[i] = softmax_output[i];         // derivative: p_i for incorrect classes
        }
    }
}

void initializeC1Weights(float c1_weight[6][5][5], float c1_bias[6]) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-0.5f, 0.5f);

    for (int i = 0; i < 6; i++) {
        c1_bias[i] = dis(gen);
        for (int j = 0; j < 5; j++) {
            for (int k = 0; k < 5; k++) {
                c1_weight[i][j][k] = dis(gen);
            }
        }
    }
}

void initializeS1Weights(float s1_weight[1][4][4], float s1_bias[1]) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-0.5f, 0.5f);

    s1_bias[0] = dis(gen);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            s1_weight[0][i][j] = dis(gen);
        }
    }
}

// Initialize fully connected layer weights (flattened input: 6*6*6 = 216 -> 10 outputs)
void initializeFWeights(float f_weight[10][216], float f_bias[10]) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-0.5f, 0.5f);

    for (int i = 0; i < 10; i++) {
        f_bias[i] = dis(gen);
        for (int j = 0; j < 216; j++) {  // 6*6*6 = 216 flattened inputs
            f_weight[i][j] = dis(gen);
        }
    }
}

// Forward pass functions
void fp_c1(const float input[28][28], float c1_preact[6][24][24], float c1_weight[6][5][5], float c1_bias[6]) {
    // Convolution: 28x28 input with 6 filters of 5x5 -> 6x24x24 output
    // Parallelize across feature maps (filters)
    #pragma omp parallel for
    for (int feature = 0; feature < 6; feature++) {
        for (int row = 0; row < 24; row++) {
            for (int col = 0; col < 24; col++) {
                float sum = c1_bias[feature];
                for (int i = 0; i < 5; i++) {
                    for (int j = 0; j < 5; j++) {
                        sum += input[row + i][col + j] * c1_weight[feature][i][j];
                    }
                }
                c1_preact[feature][row][col] = sum;
            }
        }
    }
}

void applyActivationC1(float c1_preact[6][24][24], float c1_output[6][24][24]) {
    // Parallelize across feature maps for activation
    #pragma omp parallel for collapse(2)
    for (int feature = 0; feature < 6; feature++) {
        for (int row = 0; row < 24; row++) {
            for (int col = 0; col < 24; col++) {
                c1_output[feature][row][col] = relu_function(c1_preact[feature][row][col]);
            }
        }
    }
}

void fp_s1(float c1_output[6][24][24], float s1_preact[6][6][6], float s1_weight[1][4][4], float s1_bias[1]) {
    // Subsampling/Pooling: 6x24x24 -> 6x6x6 (4x4 pooling with stride 4)
    // Parallelize across feature maps
    #pragma omp parallel for
    for (int feature = 0; feature < 6; feature++) {
        for (int row = 0; row < 6; row++) {
            for (int col = 0; col < 6; col++) {
                float sum = s1_bias[0];
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        sum += c1_output[feature][row * 4 + i][col * 4 + j] * s1_weight[0][i][j];
                    }
                }
                s1_preact[feature][row][col] = sum;
            }
        }
    }
}

void applyActivationS1(float s1_preact[6][6][6], float s1_output[6][6][6]) {
    // Parallelize activation computation
    #pragma omp parallel for collapse(2)
    for (int feature = 0; feature < 6; feature++) {
        for (int row = 0; row < 6; row++) {
            for (int col = 0; col < 6; col++) {
                s1_output[feature][row][col] = relu_function(s1_preact[feature][row][col]);
            }
        }
    }
}

void fp_preact_f(float s1_output[6][6][6], float f_preact[10], float f_weight[10][216]) {
    // Efficiently flatten s1_output and compute fully connected layer
    // Parallelize across output neurons
    #pragma omp parallel for
    for (int output_neuron = 0; output_neuron < 10; output_neuron++) {
        float sum = 0.0f;
        
        // Direct flattened indexing: feature*36 + row*6 + col
        for (int feature = 0; feature < 6; feature++) {
            for (int row = 0; row < 6; row++) {
                for (int col = 0; col < 6; col++) {
                    int flatten_idx = feature * 36 + row * 6 + col;  // 6*6 = 36
                    sum += s1_output[feature][row][col] * f_weight[output_neuron][flatten_idx];
                }
            }
        }
        f_preact[output_neuron] = sum;
    }
}

void fp_bias_f(float f_preact[10], float f_bias[10]) {
    for (int i = 0; i < 10; i++) {
        f_preact[i] += f_bias[i];
    }
}

void applyActivationF(float f_preact[10], float f_output[10]) {
    // Apply softmax activation for output layer
    softmax_function(f_preact, f_output);
}

// Backward propagation functions
void bp_weight_f(float d_f_weight[10][216], float d_f_preact[10], float s1_output[6][6][6]) {
    // Parallelize across output neurons
    #pragma omp parallel for
    for (int output_neuron = 0; output_neuron < 10; output_neuron++) {
        for (int feature = 0; feature < 6; feature++) {
            for (int row = 0; row < 6; row++) {
                for (int col = 0; col < 6; col++) {
                    int flatten_idx = feature * 36 + row * 6 + col;
                    d_f_weight[output_neuron][flatten_idx] += d_f_preact[output_neuron] * s1_output[feature][row][col];
                }
            }
        }
    }
}

void bp_bias_f(float f_bias[10], float d_f_preact[10]) {
    for (int i = 0; i < 10; i++) {
        f_bias[i] -= lr * d_f_preact[i];
    }
}

void bp_output_s1(float d_s1_output[6][6][6], float f_weight[10][216], float d_f_preact[10]) {
    // Restructure to avoid atomic operations - parallelize over output locations instead
    #pragma omp parallel for collapse(3)
    for (int feature = 0; feature < 6; feature++) {
        for (int row = 0; row < 6; row++) {
            for (int col = 0; col < 6; col++) {
                float sum = 0.0f;
                int flatten_idx = feature * 36 + row * 6 + col;
                // Serial accumulation - no race condition
                for (int output_neuron = 0; output_neuron < 10; output_neuron++) {
                    sum += f_weight[output_neuron][flatten_idx] * d_f_preact[output_neuron];
                }
                d_s1_output[feature][row][col] = sum;  // Each thread writes to unique location
            }
        }
    }
}

void bp_preact_s1(float d_s1_preact[6][6][6], float d_s1_output[6][6][6], float s1_preact[6][6][6]) {
    // Parallelize ReLU derivative computation
    #pragma omp parallel for collapse(2)
    for (int feature = 0; feature < 6; feature++) {
        for (int row = 0; row < 6; row++) {
            for (int col = 0; col < 6; col++) {
                // ReLU derivative: 1 if x > 0, else 0
                d_s1_preact[feature][row][col] = d_s1_output[feature][row][col] * relu_derivative(s1_preact[feature][row][col]);
            }
        }
    }
}

void bp_weight_s1(float d_s1_weight[1][4][4], float d_s1_preact[6][6][6], float c1_output[6][24][24]) {
    for (int feature = 0; feature < 6; feature++) {
        for (int row = 0; row < 6; row++) {
            for (int col = 0; col < 6; col++) {
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        d_s1_weight[0][i][j] += d_s1_preact[feature][row][col] * c1_output[feature][row * 4 + i][col * 4 + j];
                    }
                }
            }
        }
    }
}

void bp_bias_s1(float s1_bias[1], float d_s1_preact[6][6][6]) {
    float grad_sum = 0.0f;
    for (int feature = 0; feature < 6; feature++) {
        for (int row = 0; row < 6; row++) {
            for (int col = 0; col < 6; col++) {
                grad_sum += d_s1_preact[feature][row][col];
            }
        }
    }
    s1_bias[0] -= lr * grad_sum;
}

void bp_output_c1(float d_c1_output[6][24][24], float s1_weight[1][4][4], float d_s1_preact[6][6][6]) {
    // Backpropagate through pooling - parallelize across features
    #pragma omp parallel for
    for (int feature = 0; feature < 6; feature++) {
        for (int s_row = 0; s_row < 6; s_row++) {
            for (int s_col = 0; s_col < 6; s_col++) {
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        d_c1_output[feature][s_row * 4 + i][s_col * 4 + j] += s1_weight[0][i][j] * d_s1_preact[feature][s_row][s_col];
                    }
                }
            }
        }
    }
}

void bp_preact_c1(float d_c1_preact[6][24][24], float d_c1_output[6][24][24], float c1_preact[6][24][24]) {
    // Parallelize ReLU derivative computation
    #pragma omp parallel for collapse(2)
    for (int feature = 0; feature < 6; feature++) {
        for (int row = 0; row < 24; row++) {
            for (int col = 0; col < 24; col++) {
                // ReLU derivative: 1 if x > 0, else 0
                d_c1_preact[feature][row][col] = d_c1_output[feature][row][col] * relu_derivative(c1_preact[feature][row][col]);
            }
        }
    }
}

void bp_weight_c1(float d_c1_weight[6][5][5], float d_c1_preact[6][24][24], const float input[28][28]) {
    // Parallelize across features for weight gradient computation
    #pragma omp parallel for
    for (int feature = 0; feature < 6; feature++) {
        for (int row = 0; row < 24; row++) {
            for (int col = 0; col < 24; col++) {
                for (int i = 0; i < 5; i++) {
                    for (int j = 0; j < 5; j++) {
                        d_c1_weight[feature][i][j] += d_c1_preact[feature][row][col] * input[row + i][col + j];
                    }
                }
            }
        }
    }
}

void bp_bias_c1(float c1_bias[6], float d_c1_preact[6][24][24]) {
    for (int feature = 0; feature < 6; feature++) {
        float grad_sum = 0.0f;
        for (int row = 0; row < 24; row++) {
            for (int col = 0; col < 24; col++) {
                grad_sum += d_c1_preact[feature][row][col];
            }
        }
        c1_bias[feature] -= lr * grad_sum;
    }
}

// Weight update functions
void updateFWeights(float f_weight[10][216], float d_f_weight[10][216]) {
    // Parallelize weight updates
    #pragma omp parallel for
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 216; j++) {
            f_weight[i][j] -= lr * d_f_weight[i][j];
        }
    }
}

void updateS1Weights(float s1_weight[1][4][4], float d_s1_weight[1][4][4]) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            s1_weight[0][i][j] -= lr * d_s1_weight[0][i][j];
        }
    }
}

void updateC1Weights(float c1_weight[6][5][5], float d_c1_weight[6][5][5]) {
    // Parallelize weight updates across filters
    #pragma omp parallel for
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 5; j++) {
            for (int k = 0; k < 5; k++) {
                c1_weight[i][j][k] -= lr * d_c1_weight[i][j][k];
            }
        }
    }
}

#endif // LAYER_H
