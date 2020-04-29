#include <stdio.h>
#include <cuda.h>
#include "tensor.h"
#include "sum.cuh"

const int BLOCK_SIZE = 1024;

__global__ void mul_in_place(float *a, float* b, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
    {
        a[index] = a[index] * b[index];
    }
}

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

    void cross_entropy(TENSOR *a, TENSOR* b, TENSOR *target) {
        float* gpu_a;
        int size = a->mat_shape.x * a->mat_shape.y;
        int msize = size * sizeof(float);
        cudaMalloc((void**)&gpu_a, msize);
        cudaMemcpy(gpu_a, &a->data[0], msize, cudaMemcpyHostToDevice);

        float* gpu_b;
        cudaMalloc((void**)&gpu_b, msize);
        cudaMemcpy(gpu_b, &b->data[0], msize, cudaMemcpyHostToDevice);

        float* gpu_target;
        int msize_target = sizeof(float);
        cudaMalloc((void**)&gpu_target, msize_target);

        // MUL

        dim3 blockSize = dim3(BLOCK_SIZE);
        dim3 gridSize = dim3((size + BLOCK_SIZE - 1) / BLOCK_SIZE);
        mul_in_place<<<gridSize, blockSize>>>(gpu_a, gpu_b, a->mat_shape.x * a->mat_shape.y);

        // SUM1 - LOG - NEG

        float* gpu_sum1_target;
        int msize_sum1_target = a->mat_shape.x * sizeof(float);
        cudaMalloc((void**)&gpu_sum1_target, msize_sum1_target);

        gridSize = dim3((a->mat_shape.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
        sum1_log_neg<<<gridSize, blockSize>>>(gpu_a, gpu_sum1_target, a->mat_shape.y, a->mat_shape.x);

        // SUM0

        gridSize = dim3((a->mat_shape.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
        sum0<<<gridSize, blockSize>>>(gpu_sum1_target, gpu_target, 1, a->mat_shape.x);

        // copy back

        cudaMemcpy(&target->data[0], gpu_target, msize_target, cudaMemcpyDeviceToHost);
        target->data[0] /= a->mat_shape.x;

        cudaFree(gpu_a);
        cudaFree(gpu_b);
        cudaFree(gpu_sum1_target);
        cudaFree(gpu_target);
    }

}

