#include <stdio.h>
#include <cuda.h>
#include "tensor.h"

const int BLOCK_SIZE = 32;

__global__ void linear(float *a, float *x, float *b, float *c, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if (col < k && row < m)
    {
        for (int i = 0; i < n; i++)
        {
            sum += a[row * n + i] * x[i * k + col];
        }
        c[row * k + col] = sum + b[col];
    }
}

extern "C" {

    void linear(const TENSOR *a, const TENSOR *x, const TENSOR* b, TENSOR *target)
    {
        float* gpu_a;
        size_t size = a->shapeX * a->shapeY * sizeof(float);
        cudaMalloc((void**)&gpu_a, size);
        cudaMemcpy(gpu_a, &a->data[0], size, cudaMemcpyHostToDevice);

        float* gpu_b;
        size_t size_b = b->shapeY * sizeof(float);
        cudaMalloc((void**)&gpu_b, size_b);
        cudaMemcpy(gpu_b, &b->data[0], size_b, cudaMemcpyHostToDevice);

        float* gpu_x;
        size = x->shapeX * x->shapeY * sizeof(float);
        cudaMalloc((void**)&gpu_x, size);
        cudaMemcpy(gpu_x, &x->data[0], size, cudaMemcpyHostToDevice);

        float* gpu_target;
        size = target->shapeX * target->shapeY * sizeof(float);
        cudaMalloc(&gpu_target, size);

        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridSize((x->shapeY + BLOCK_SIZE - 1) / BLOCK_SIZE, (a->shapeX + BLOCK_SIZE - 1) / BLOCK_SIZE);
        linear<<<gridSize, blockSize>>>(gpu_a, gpu_x, gpu_b, gpu_target, a->shapeX, a->shapeY, x->shapeY);

        cudaMemcpy(&target->data[0], gpu_target, size, cudaMemcpyDeviceToHost);

        cudaFree(gpu_a);
        cudaFree(gpu_x);
        cudaFree(gpu_b);
        cudaFree(gpu_target);
    }

}

