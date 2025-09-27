#include <vector>
#include <random>
#include <fstream>
#include <iomanip>
#include <iostream>

//Since MNIST files are in special binary format, we can skip opencv and stuff and use c++ only.

// We have 2d images and their labels
struct MNISTData {
    std::vector<std::vector<float>> images;
    std::vector<int> labels;
    int num_images;
    int image_size;
};

// Byte stream from MNIST is big-endian, need to convert to little-endian for using it since this is what PC uses.
int reverseInt(int i) {
    unsigned char c1, c2,c3, c4;
    c1 = i & 255; //Get the last 8 bits
    c2 = (i >> 8) & 255; //Get the 2nd byte
    c3 = (i >> 16) & 255; //Get the 3rd byte
    c4 = (i >> 24) & 255; //Get the 4th byte
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4; //Reverse the order of bytes
}


// Load MNIST dataset
MNISTData loadMNISTImages(const std::string &filename) {
    MNISTData data;
    std::ifstream file(filename, std::ios::binary);
    if(!file.is_open()){
        std::cerr << "Error: Could not open " << filename << "\n";
        std::cerr << "Please ensure the MNIST files are in the current directory." << std::endl;
        data.num_images = 0;
        return data;
    }

    int magic_number = 0;
    int number_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;

    // MNIST files start with 4 bytes magic number followed by number of images, rows and columns
    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);

    file.read((char*)&number_of_images, sizeof(number_of_images));
    number_of_images = reverseInt(number_of_images);

    file.read((char*)&n_rows, sizeof(n_rows));
    n_rows = reverseInt(n_rows);

    file.read((char*)&n_cols, sizeof(n_cols));
    n_cols = reverseInt(n_cols);

    data.num_images = number_of_images;
    data.image_size = n_rows * n_cols;
    data.images.resize(number_of_images);

    // row*col gives the starting of the row, c is the column offset. this is a flattened 1D array representation of 2D image
    for(int i = 0; i < number_of_images; i++) {
        data.images[i].resize(data.image_size);
        // MNIST images have one byte per pixel. can be negative as well.
        for(int r = 0; r < n_rows; r++) {
            for(int c = 0; c < n_cols; c++) {
                unsigned char temp = 0;
                file.read((char*)&temp, sizeof(temp));
                data.images[i][r * n_cols + c] = static_cast<float>(temp) / 255.0f; // Normalize pixel values to [0, 1]
            }
        }
    }

    file.close();
    return data;
}


// Load MNIST labels
std::vector<int> loadMNISTLabels(const std::string &filename) {
    std::vector<int> labels;
    std::ifstream file(filename, std::ios::binary);

    if(!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << "\n";
        return labels;
    }

    int magic_number = 0;
    int number_of_labels = 0;

    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);

    file.read((char*)&number_of_labels, sizeof(number_of_labels));
    number_of_labels = reverseInt(number_of_labels);

    labels.resize(number_of_labels);
    for(int i = 0; i < number_of_labels; i++) {
        unsigned char temp = 0;
        file.read((char*)&temp, sizeof(temp));
        labels[i] = static_cast<int>(temp);
    }

    file.close();
    return labels;
}

