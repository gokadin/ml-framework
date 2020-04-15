#include <stdio.h>
#include <cuda.h>
#include "tensor.h"

const int BLOCK_SIZE = 32;

__global__ void matmul(float *a, float *b, float *c, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if (col < k && row < m)
    {
        for (int i = 0; i < n; i++)
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}

extern "C" {

    void matmul(const TENSOR *a, const TENSOR* b, TENSOR *target) {
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

        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridSize((b->shapeY + BLOCK_SIZE - 1) / BLOCK_SIZE, (a->shapeX + BLOCK_SIZE - 1) / BLOCK_SIZE);
        matmul<<<gridSize, blockSize>>>(gpu_a, gpu_b, gpu_target, a->shapeX, a->shapeY, b->shapeY);

        cudaMemcpy(&target->data[0], gpu_target, size, cudaMemcpyDeviceToHost);

        cudaFree(gpu_a);
        cudaFree(gpu_b);
        cudaFree(gpu_target);
    }

}

