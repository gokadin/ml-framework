#include <stdio.h>
#include <cuda.h>
#include "tensor.h"

const int BLOCK_SIZE = 1024;

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

__global__ void sum1(float *a, float *target, int width, int height)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if (row < height)
    {
        for (int i = 0; i < width; i++)
        {
            sum += a[row * width + i];
        }
        target[row] = sum;
    }
}

extern "C" {

    void sum(TENSOR *a, int axis, TENSOR *target) {
        float* gpu_a;
        size_t size = a->mat_shape->x * a->mat_shape->y * sizeof(float);
        cudaMalloc((void**)&gpu_a, size);
        cudaMemcpy(gpu_a, &a->data[0], size, cudaMemcpyHostToDevice);

        float* gpu_target;
        size_t size_target;

        if (axis == 0) {
            size_target = a->mat_shape->y * sizeof(float);
            cudaMalloc((void**)&gpu_target, size_target);

            dim3 blockSize(BLOCK_SIZE);
            dim3 gridSize((a->mat_shape->y + BLOCK_SIZE - 1) / BLOCK_SIZE);
            sum0<<<gridSize, blockSize>>>(gpu_a, gpu_target, a->mat_shape->y, a->mat_shape->x);
        } else if (axis == 1) {
            size_target = a->mat_shape->x * sizeof(float);
            cudaMalloc((void**)&gpu_target, size_target);

            dim3 blockSize(BLOCK_SIZE);
            dim3 gridSize((a->mat_shape->x + BLOCK_SIZE - 1) / BLOCK_SIZE);
            sum1<<<gridSize, blockSize>>>(gpu_a, gpu_target, a->mat_shape->y, a->mat_shape->x);
        }

        cudaMemcpy(&target->data[0], gpu_target, size_target, cudaMemcpyDeviceToHost);

        cudaFree(gpu_a);
        cudaFree(gpu_target);
    }

}

