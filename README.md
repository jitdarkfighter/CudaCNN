# CudaCNN
Implementation of CNN in C++ for MNIST digit classification.

## Architecture
A simple CNN with the following layers:
- **Conv2D Layer**: 1→6 channels, 5×5 kernel (156 parameters)
- **ReLU Activation**
- **Conv2D Layer** (used as pooling): 6→6 channels, 4×4 kernel, stride 4 (486 parameters) 
- **ReLU Activation**
- **Flatten Layer**
- **Linear Layer**: 216→10 (2170 parameters)
- **Softmax Activation** (for output probabilities)
- **Total**: ~2812 parameters

## Quick Start Instructions

### Prerequisites
- G++ compiler with C++11 support
- OpenMP support (usually included with GCC)
- MNIST dataset files (already included in project root)

### Files Structure
```
├── train-images-idx3-ubyte    # Training images (60,000 samples)
├── train-labels-idx1-ubyte    # Training labels
├── t10k-images-idx3-ubyte     # Test images (10,000 samples)
├── t10k-labels-idx1-ubyte     # Test labels
├── Sequential/                 # Sequential implementation
└── OpenMP/                    # Parallel implementation
```

### Running the Code

#### Sequential Version
```bash
cd Sequential
g++ -o main main.cpp -std=c++11 -fopenmp
./main
```

#### OpenMP Parallel Version
```bash
cd OpenMP
g++ -o main main.cpp -std=c++11 -fopenmp
./main
```

### Modifying Hyperparameters

You can easily modify training parameters by editing the constants in `main.cpp`:

#### Sequential Version (`Sequential/main.cpp`)
```cpp
// Training parameters (around line 47)
const int epochs = 5;           // Number of training epochs
const int batch_size = 100;     // Batch size for training
```

#### OpenMP Version (`OpenMP/main.cpp`)
```cpp
// Training parameters (top of file)
const int EPOCHS = 5;           // Number of training epochs
const int BATCH_SIZE = 100;     // Batch size for training
const int NUM_THREADS = 16;     // Number of OpenMP threads
```

### Expected Output
The program will:
1. Load MNIST dataset (60,000 training + 10,000 test images)
2. Train the CNN for specified epochs
3. Display training progress with loss and accuracy
4. Show test accuracy after each epoch
5. Save the trained model as `smolCNN.bin`
6. Display total training time

### Performance Comparison
- **Sequential**: Runs on single thread
- **OpenMP**: Utilizes multiple threads for faster training

### Original PyTorch Implementation Reference
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class smolCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.pool = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=4, stride=4)
        self.fc = nn.Linear(6 * 6 * 6, 10)

    def forward(self, x):
        x = self.conv1(x)  # (batch_size, 6, 24, 24)
        x = F.relu(x)
        x = self.pool(x)   # (batch_size, 6, 6, 6)
        x = F.relu(x)
        x = x.view(x.size(0), -1)  # (batch_size, 216)
        x = self.fc(x)     # (batch_size, 10)
        return x
```


## Example Run Commands

**Quick test with 2 epochs:**
```bash
# Edit main.cpp to set epochs = 2
cd Sequential
g++ -o main main.cpp -std=c++11 -fopenmp && ./main
```

**Parallel version with custom thread count:**
```bash
# Edit main.cpp to set NUM_THREADS = 8
cd OpenMP  
g++ -o main main.cpp -std=c++11 -fopenmp && ./main
```

