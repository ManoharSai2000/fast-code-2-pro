# Parallelization of CNN - Alexnet: CPU and GPU

## OpenMP
The following files correspond to our OpenMP implementation of the network.

```
./AlexNet/src/activation_layer.c
./AlexNet/src/batchnorm_layer.c
./AlexNet/src/convolution_layer.c
./AlexNet/src/dropout_layer.c
./AlexNet/src/matrix.c
./AlexNet/src/maxpooling_layer.c
```

## CUDA
The following files correspond to our CUDA implementation of the network.
```
./cuda/src/alexnet.cpp
```
We run the following command to compile 
```nvcc -x cu src/alexnet.cpp src/NetworkModel.cpp src/FullyConnected.cpp src/Sigmoid.cpp src/Dropout.cpp src/SoftmaxClassifier.cpp src/MNISTDataLoader.cpp src/ReLU.cpp src/Tensor.cpp src/Conv2d.cpp src/MaxPool.cpp src/LinearLRScheduler.cpp -I../include -o alexnet.x -arch=sm_70 -std=c++11```

License
----

MIT

