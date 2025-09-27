Images are usually in Big-endian.
The first 32-bit integer at the very begining of a binary file(both images and labels) are called magic numers
Images have a magic number os 2051 `0x00000803`
Text or labels have 2049 `0x00000801`

The MNIST files are stored in big-endian format (most significant byte first), but most PCs (like x86) use little-endian. So we need to use reverseInt function to convert it into the system's native integer order.

Neural networks usually store images as flat arrays in memory for efficiency.
Convolution loops just convert 1D → 2D indices on the fly.