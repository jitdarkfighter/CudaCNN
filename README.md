# CudaCNN
Implementation of CNN in C++.


#### Pytorch Implementation
# CudaCNN
Implementation of CNN in C++.

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


#### Original PyTorch Implementation
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
        # Input: (batch_size, 1, 28, 28)
        # Convolution (28x28 -> 24x24)
        x = self.conv1(x)  # shape: (batch_size, 6, 24, 24)
        x = F.relu(x)
        # Pooling (custom conv pooling: 24x24 -> 6x6)
        x = self.pool(x)   # shape: (batch_size, 6, 6, 6)
        x = F.relu(x)
        # Flatten
        x = x.view(x.size(0), -1)  # (batch_size, 6*6*6)
        # Fully connected
        x = self.fc(x)  # (batch_size, 10)
        return x
```

**Parameter Count:**
- Conv layer: 6*(5*5+1) = 156 parameters
- Pooling Layer: 6*(6*(4*4+1)) = 486 parameters  
- FC layer: 216*10 + 10 = 2170 parameters
- **Total**: ~2812 parameters

