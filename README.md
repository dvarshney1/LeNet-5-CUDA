# LeNet-5-CUDA
A modified LeNet-5 CNN architecture using the mini-DNN framework in CUDA

## CUDA Optimizations to Improve Forward Inference:
- Unroll + shared-memory Matrix multiply
- Shared memory convolution
- Weight values in constant memory
- CUDA streams with pinned memory
- Restrict and loop unrolling
- Fixed point (FP16) arithmetic
- Multiple kernel implementations

## Future Updates to Further Imporve Performance
- Tensor Cores
- Register-tiled matrix multiplcation
