Since im making a CNN from scratch, I've decided to go with the OG LeNet architecture.
One key difference between that and normal CNN is that LeNet has a subsampling layer(normally called pooling) which has it's own learnable parameters.


### dataloader.h
Images are usually in Big-endian.
The first 32-bit integer at the very begining of a binary file(both images and labels) are called magic numers
Images have a magic number os 2051 `0x00000803`
Text or labels have 2049 `0x00000801`

The MNIST files are stored in big-endian format (most significant byte first), but most PCs (like x86) use little-endian. So we need to use reverseInt function to convert it into the system's native integer order.

Neural networks usually store images as flat arrays in memory for efficiency.
Convolution loops just convert 1D → 2D indices on the fly.


### layers.h

Specs

| Layer | Input Size | Weight Size | Bias Size | Output Size |
| ----- | ---------- | ----------- | --------- | ----------- |
| C1    | 28×28      | 6 × 5×5     | 6         | 6 × 24×24   |
| S1    | 6 × 24×24  | 1 × 4×4     | 1         | 6 × 6×6     |
| F     | 6 × 6×6    | 10 × 6×6×6  | 10        | 10          |

### Activation Functions & Learning Rate Impact

Initially implemented the network using sigmoid activation functions throughout, which is historically accurate to early CNN architectures. However, modern practice suggests using ReLU for hidden layers and softmax for multi-class output. After switching from sigmoid to ReLU (hidden layers) and softmax (output layer), the model initially performed poorly with very low accuracy (~18%) and high loss values (~2.3). The critical insight was that ReLU networks require a lower learning rate compared to sigmoid networks. Reducing the learning rate from 0.1 to 0.01 dramatically improved performance, achieving 99.70% training accuracy and 96.80% test accuracy within 5 epochs. This demonstrates the importance of hyperparameter tuning when changing activation functions, as ReLU's non-saturating gradients can cause instability with learning rates that work fine for sigmoid.
