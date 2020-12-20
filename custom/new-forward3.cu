#include <cmath>
#include <iostream>
#include <stdio.h>
#include <pthread.h>
#include <time.h>
#include "gpu-new-forward.h"
#include "cuda_fp16.h"

#define TILE_WIDTH 16
#define K_COMMON 7
#define C_LAYER1 1
#define M_LAYER1 6
#define C_LAYER2 6
#define M_LAYER2 16
#define TILE_WIDTH_1D 512 //GPU implementation
#define NUM_PTHREADS 32


__global__ void conv_forward_kernel_layer1(float *y, const float * x, const half* k_layer1, const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    // (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)W_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k_layer1[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#define loc_mem(i2, i1, i0) loc_mem1[(i2) * ((TILE_WIDTH+K_COMMON) * (TILE_WIDTH+K_COMMON)) + (i1) * (TILE_WIDTH+K_COMMON) + (i0)]

    // Insert your GPU convolution kernel code here
    int W_grid = ceil((1.0 * W_out)/TILE_WIDTH);
    int H_grid = ceil((1.0 * H_out)/TILE_WIDTH);

    int b,m,h,w;
    b = blockIdx.x;
    m = blockIdx.y;
    h = (blockIdx.z/W_grid)*TILE_WIDTH + threadIdx.y;
    w = (blockIdx.z % W_grid)*TILE_WIDTH + threadIdx.x;

    __shared__ half loc_mem1[C_LAYER1 * (TILE_WIDTH + K_COMMON) * (TILE_WIDTH + K_COMMON)];

    loc_mem(0, threadIdx.y, threadIdx.x) = __float2half(x4d(b, 0, h, w));
    loc_mem(0, threadIdx.y + (K-1), threadIdx.x) = __float2half(x4d(b, 0, h + (K - 1), w));
    loc_mem(0, threadIdx.y, threadIdx.x + (K - 1)) = __float2half(x4d(b, 0, h, w + (K - 1)));
    loc_mem(0, threadIdx.y + (K - 1), threadIdx.x + (K - 1)) = __float2half(x4d(b, 0, h + (K - 1), w + (K - 1)));

    __syncthreads();
   
    half acc = 0.0f;
    for (int p = 0; p < K; p++) {
        for (int q = 0; q < K; q++) {
            acc = acc + loc_mem(0, threadIdx.y+p, threadIdx.x+q) * k4d(m, 0, p, q); // C = C_LAYER1 - 1
        }
    }
    y4d(b, m, h, w) = __half2float(acc);

#undef y4d
#undef x4d
#undef k4d
#undef loc_mem
}

__global__ void conv_forward_kernel_layer2(float *y, const float * x, const half* k_layer2, const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    // (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)W_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k_layer2[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#define loc_mem(i2, i1, i0) loc_mem2[(i2) * ((TILE_WIDTH+K) * (TILE_WIDTH+K)) + (i1) * (TILE_WIDTH+K) + (i0)]

    // Insert your GPU convolution kernel code here
    int W_grid = ceil((1.0 * W_out)/TILE_WIDTH);
    int H_grid = ceil((1.0 * H_out)/TILE_WIDTH);

    int b,m,h,w;
    b = blockIdx.x;
    m = blockIdx.y;
    h = (blockIdx.z/W_grid)*TILE_WIDTH + threadIdx.y;
    w = (blockIdx.z % W_grid)*TILE_WIDTH + threadIdx.x;

    __shared__ half loc_mem2[C_LAYER2 * (TILE_WIDTH + K_COMMON) * (TILE_WIDTH + K_COMMON)]; 

    for(int c = 0; c < C; c++) {
      loc_mem(c, threadIdx.y, threadIdx.x) = __float2half(x4d(b, c, h, w));
      loc_mem(c, threadIdx.y + (K-1), threadIdx.x) = __float2half(x4d(b, c, h + (K - 1), w));
      loc_mem(c, threadIdx.y, threadIdx.x + (K - 1)) = __float2half(x4d(b, c, h, w + (K - 1)));
      loc_mem(c, threadIdx.y + (K - 1), threadIdx.x + (K - 1)) = __float2half(x4d(b, c, h + (K - 1), w + (K - 1)));
    }

    __syncthreads();

    if (h < H_out && w < W_out) {
        half acc = 0.0f;
        for (int c = 0; c < C; c++) {
            for (int p = 0; p < K; p++) {
                for (int q = 0; q < K; q++) {
                    acc = acc + (loc_mem(c, threadIdx.y+p, threadIdx.x+q) * k4d(m, c, p, q));
                }
            }
        }
        y4d(b, m, h, w) = __half2float(acc);
    }

#undef y4d
#undef x4d
#undef k4d
#undef loc_mem
}

struct args {
  const float *f_arr_k;
  half *h_arr_k;
  int len_k;
  int start_idx_k;
};


__host__ void * f2h_host(void *arg_list){
  int len_k = ((struct args *)arg_list)->len_k;
  for(int i = 0; i < ceil((1.0*len_k)/(NUM_PTHREADS)); i++){
    int idx  = ((struct args *)arg_list)->start_idx_k + i;
    if(idx >= len_k)
      break;
    (((struct args *)arg_list)->h_arr_k)[idx] = __float2half((((struct args *)arg_list)->f_arr_k)[idx]);
  }
  return NULL;
}


__host__ void GPUInterface::conv_forward_gpu3(float *host_y, const float *host_x, const float *host_k, const int B,
                                            const int M, const int C, const int H, const int W, const int K)
{

    float *device_y;
    float *device_x;
    half *k_layer1;
    half *k_layer2;
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int x_len = (W * H * C * B);
    int y_len = (H_out * W_out * M * B);
    int k_len = (K * K * M * C);

    half *host_k16 = (half *)malloc(k_len * sizeof(half));
    
    pthread_t tids[NUM_PTHREADS];
    struct args arg_list[NUM_PTHREADS];

    for (unsigned int i = 0; i <  NUM_PTHREADS; i++) {
        arg_list[i].len_k = k_len;
        arg_list[i].f_arr_k = host_k;
        arg_list[i].h_arr_k = host_k16;
        arg_list[i].start_idx_k = i*ceil((1.0*k_len)/NUM_PTHREADS);
        pthread_create(tids + i, NULL, f2h_host, (void *)(arg_list + i));
    }

    for (unsigned int i = 0; i <  NUM_PTHREADS; i++) {
      pthread_join(tids[i], NULL);
    }

    cudaMalloc((void **) &device_y, y_len * sizeof(float));
    cudaMalloc((void **) &device_x, x_len  * sizeof(float));

    int W_grid = ceil((1.0 * W_out)/TILE_WIDTH);
    int H_grid = ceil((1.0 * H_out)/TILE_WIDTH);
    int Z = W_grid * H_grid;

    dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 DimGrid(B, M, Z);

    if (C == 1) { 
        cudaMalloc((void **) &k_layer1, k_len  * sizeof(half));
        cudaMemcpy(k_layer1, host_k16, k_len * sizeof(half), cudaMemcpyHostToDevice); 
    }
    else { 
        cudaMalloc((void **) &k_layer2, k_len  * sizeof(half));
        cudaMemcpy(k_layer2, host_k16, k_len * sizeof(half), cudaMemcpyHostToDevice); 
    }

    cudaMemcpy(device_x, host_x, x_len * sizeof(float), cudaMemcpyHostToDevice);

    if (C == 1) { conv_forward_kernel_layer1<<<DimGrid, DimBlock>>> (device_y, device_x, k_layer1, B, M, C, H, W, K); }
    else { conv_forward_kernel_layer2<<<DimGrid, DimBlock>>> (device_y, device_x, k_layer2, B, M, C, H, W, K); }

    // Copy the output back to host
    cudaMemcpy(host_y, device_y, y_len * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(device_y);
    cudaFree(device_x);

    if(C == 1) {
        cudaFree(k_layer1);
    } else {
        cudaFree(k_layer2);
    }

    free(host_k16);
}

__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
