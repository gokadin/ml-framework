#include <stdio.h>
#include <cuda.h>
#include "cudautils.h"
#include "tensor.h"
#include "sum.cuh"

const int BLOCK_SIZE = 1024;

__global__ void sum1_log_neg(float *a, float *target, int width, int height)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if (row < height)
    {
        for (int i = 0; i < width; i++)
        {
            sum += a[row * width + i];
        }
        target[row] = -log(sum);
    }
}

extern "C" {

    __declspec(dllexport) void gpu_bce_forward(TENSOR *target, TENSOR *a, TENSOR *b) {
        float* gpu_a;
        size_t a_size = a->mat_size * sizeof(float);
        checkCudaErr(cudaMalloc((void**)&gpu_a, a_size));
        checkCudaErr(cudaMemcpy(gpu_a, &a->data[0], a_size, cudaMemcpyHostToDevice));

        float* gpu_b;
        size_t b_size = b->mat_size * sizeof(float);
        checkCudaErr(cudaMalloc((void**)&gpu_b, b_size));
        checkCudaErr(cudaMemcpy(gpu_b, &b->data[0], b_size, cudaMemcpyHostToDevice));

        float* gpu_target;
        size_t target_size = target->mat_size * sizeof(float);
        checkCudaErr(cudaMalloc((void**)&gpu_target, target_size));

        // SUM1 - LOG - NEG

        float* gpu_sum1_target;
        size_t sum1_target_size = a->mat_shape[0] * sizeof(float);
        checkCudaErr(cudaMalloc((void**)&gpu_sum1_target, sum1_target_size));

        dim3 blockSize = dim3(BLOCK_SIZE, 1);
        dim3 gridSize = dim3((a->mat_shape[0] + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
        sum1_log_neg<<<gridSize, blockSize>>>(gpu_a, gpu_sum1_target, a->mat_shape[1], a->mat_shape[0]);
        checkCudaKernelErr("sum1_log_neg", blockSize, gridSize);

        // SUM0

        sum0<<<gridSize, blockSize>>>(gpu_sum1_target, gpu_target, 1, a->mat_shape[0]);
        checkCudaKernelErr("sum0", blockSize, gridSize);

        // copy back

        checkCudaErr(cudaMemcpy(&target->data[0], gpu_target, target_size, cudaMemcpyDeviceToHost));
        target->data[0] /= a->mat_shape[0];

        cudaFree(gpu_a);
        cudaFree(gpu_b);
        cudaFree(gpu_sum1_target);
        cudaFree(gpu_target);
    }

}

