#include <stdio.h>
#include <cuda.h>
#include "tensor.h"

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

__global__ void sum0(float *a, float *target, int width, int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if (col < width)
    {
        for (int i = 0; i < height; i++)
        {
            sum += a[col + i * width];
        }
        target[col] = sum;
    }
}

extern "C" {

    void cross_entropy(TENSOR *a, TENSOR* b, TENSOR *target) {
        float* gpu_a;
        int size = a->shapeX * a->shapeY;
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
        mul_in_place<<<gridSize, blockSize>>>(gpu_a, gpu_b, a->shapeX * a->shapeY);

        // SUM1 - LOG - NEG

        float* gpu_sum1_target;
        int msize_sum1_target = a->shapeX * sizeof(float);
        cudaMalloc((void**)&gpu_sum1_target, msize_sum1_target);

        gridSize = dim3((a->shapeX + BLOCK_SIZE - 1) / BLOCK_SIZE);
        sum1_log_neg<<<gridSize, blockSize>>>(gpu_a, gpu_sum1_target, a->shapeY, a->shapeX);

        // SUM0

        gridSize = dim3((a->shapeX + BLOCK_SIZE - 1) / BLOCK_SIZE);
        sum0<<<gridSize, blockSize>>>(gpu_sum1_target, gpu_target, 1, a->shapeX);

        // copy back

        cudaMemcpy(&target->data[0], gpu_target, msize_target, cudaMemcpyDeviceToHost);
        target->data[0] /= a->shapeX;

        cudaFree(gpu_a);
        cudaFree(gpu_b);
        cudaFree(gpu_sum1_target);
        cudaFree(gpu_target);
    }

}

