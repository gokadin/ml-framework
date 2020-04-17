#include <stdio.h>
#include <cuda.h>
#include "tensor.h"

const int BLOCK_SIZE = 1024;

__global__ void mul(float *a, float *b, float *c, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
    {
        c[index] = a[index] * b[index];
    }
}

extern "C" {

    void mul(const TENSOR *a, const TENSOR* b, TENSOR *target) {
        float* gpu_a;
        size_t size = a->shapeX * a->shapeY * sizeof(float);
        cudaMalloc((void**)&gpu_a, size);
        cudaMemcpy(gpu_a, &a->data[0], size, cudaMemcpyHostToDevice);

        float* gpu_b;
        size = b->shapeX * b->shapeY * sizeof(float);
        cudaMalloc((void**)&gpu_b, size);
        cudaMemcpy(gpu_b, &b->data[0], size, cudaMemcpyHostToDevice);

        float* gpu_target;
        size = target->shapeX * target->shapeY * sizeof(float);
        cudaMalloc(&gpu_target, size);

        dim3 blockSize(BLOCK_SIZE);
        dim3 gridSize((a->shapeY * a->shapeX + BLOCK_SIZE - 1) / BLOCK_SIZE);
        mul<<<gridSize, blockSize>>>(gpu_a, gpu_b, gpu_target, a->shapeX * a->shapeY);

        cudaMemcpy(&target->data[0], gpu_target, size, cudaMemcpyDeviceToHost);

        cudaFree(gpu_a);
        cudaFree(gpu_b);
        cudaFree(gpu_target);
    }

}

