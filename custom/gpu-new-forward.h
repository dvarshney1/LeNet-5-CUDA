#ifndef SRC_LAYER_GPU_NEW_FORWARD_H
#define SRC_LAYER_GPU_NEW_FORWARD_H

class GPUInterface
{
    public:
    void get_device_properties();
    void print_essentials(const int B, const int C, const int H, const int W, const int M, const int K);
    void conv_forward_gpu(float *host_y, const float *host_x, const float *host_k, const int B, const int M, const int C, const int H, const int W, const int K);
    void conv_forward_gpu1(float *host_y, const float *host_x, const float *host_k, const int B, const int M, const int C, const int H, const int W, const int K);
    void conv_forward_gpu2(float *host_y, const float *host_x, const float *host_k, const int B, const int M, const int C, const int H, const int W, const int K);
    void conv_forward_gpu3(float *host_y, const float *host_x, const float *host_k, const int B, const int M, const int C, const int H, const int W, const int K);
    void conv_forward_gpu4(float *host_y, const float *host_x, const float *host_k, const int B, const int M, const int C, const int H, const int W, const int K);
    void conv_forward_gpu5(float *host_y, const float *host_x, const float *host_k, const int B, const int M, const int C, const int H, const int W, const int K);
    void conv_forward_gpu6(float *host_y, const float *host_x, const float *host_k, const int B, const int M, const int C, const int H, const int W, const int K);
};

#endif 