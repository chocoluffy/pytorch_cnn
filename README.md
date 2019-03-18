Customized convolutional neural neworks trained with CIFAR-10 dataset.

Architecture:
- Use 64 11*11 filters for the first convolutional layer.
- followed by 2*2 max pooling layer (stride of 2). 
- The next two convolutional layers use 128 3*3 filters followed by the ReLU activation function.
- batch normalization layers added right after each convolutional layers.
