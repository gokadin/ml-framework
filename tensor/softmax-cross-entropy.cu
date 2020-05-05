#include <stdio.h>
#include <cuda.h>
#include "cudautils.h"
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

    void gpu_sce_forward(TENSOR *a, TENSOR* b, TENSOR *target) {
        float* gpu_a;
        size_t a_size = a->mat_shape->size * sizeof(float);
        checkCudaErr(cudaMalloc((void**)&gpu_a, a_size));
        checkCudaErr(cudaMemcpy(gpu_a, &a->data[0], a_size, cudaMemcpyHostToDevice));

        float* gpu_b;
        size_t b_size = b->mat_shape->size * sizeof(float);
        checkCudaErr(cudaMalloc((void**)&gpu_b, b_size));
        checkCudaErr(cudaMemcpy(gpu_b, &b->data[0], b_size, cudaMemcpyHostToDevice));

        float* gpu_target;
        size_t target_size = target->mat_shape->size * sizeof(float);
        checkCudaErr(cudaMalloc((void**)&gpu_target, target_size));

        // MUL

        dim3 blockSize = dim3(BLOCK_SIZE);
        dim3 gridSize = dim3((a->mat_shape->size + BLOCK_SIZE - 1) / BLOCK_SIZE);
        mul_in_place<<<gridSize, blockSize>>>(gpu_a, gpu_b, a->mat_shape->size);
        checkCudaKernelErr("mul_in_place", blockSize, gridSize);

        // SUM1 - LOG - NEG

        float* gpu_sum1_target;
        size_t sum1_target_size = a->mat_shape->x * sizeof(float);
        checkCudaErr(cudaMalloc((void**)&gpu_sum1_target, sum1_target_size));

        gridSize.x = (a->mat_shape->x + BLOCK_SIZE - 1) / BLOCK_SIZE;
        sum1_log_neg<<<gridSize, blockSize>>>(gpu_a, gpu_sum1_target, a->mat_shape->y, a->mat_shape->x);
        checkCudaKernelErr("sum1_log_neg", blockSize, gridSize);

        // SUM0

        gridSize.x = 1;
        sum0<<<gridSize, blockSize>>>(gpu_sum1_target, gpu_target, 1, a->mat_shape->x);
        checkCudaKernelErr("sum0", blockSize, gridSize);

        // copy back

        checkCudaErr(cudaMemcpy(&target->data[0], gpu_target, target_size, cudaMemcpyDeviceToHost));
        target->data[0] /= a->mat_shape->x;

        cudaFree(gpu_a);
        cudaFree(gpu_b);
        cudaFree(gpu_sum1_target);
        cudaFree(gpu_target);
    }

}

