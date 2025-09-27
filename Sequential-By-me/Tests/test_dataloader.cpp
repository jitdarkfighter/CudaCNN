#include "dataloader.h"
#include <iostream>

int main() {
    MNISTData train_data = loadMNISTImages("../train-images-idx3-ubyte");
    std::vector<int> train_labels = loadMNISTLabels("../train-labels-idx1-ubyte");
    
    std::cout << "Number of training images: " << train_data.num_images << std::endl;
    std::cout << "Image size: " << train_data.image_size << std::endl;
    std::cout << "Number of labels: " << train_labels.size() << std::endl;
    
    return 0;
}