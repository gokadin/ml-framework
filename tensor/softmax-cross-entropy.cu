#include <stdio.h>
#include <cuda.h>
#include "cudautils.h"
#include "tensor.h"
#include "sum.cuh"

const int BLOCK_SIZE = 1024;
const int SOFTMAX_MUL_BLOCK_SIZE = 32;

__global__ void softmax_mul(float *a, float *b, float *target, int x, int y)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < y && row < x)
    {
        float sum = 0;
        for (int j = 0; j < y; j++)
        {
            sum += exp(a[row * y + j]);
        }

        for (int j = 0; j < y; j++)
        {
            int index = row * y + j;
            a[index] = exp(a[index]) / sum;
            target[index] = a[index] * b[index];
        }
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

    void gpu_sce_forward(TENSOR *target, TENSOR *a, TENSOR *b) {
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

        // SOFTMAX - MUL

        float* gpu_softmax_mul_target;
        size_t softmax_mul_target_size = a->mat_shape->size * sizeof(float);
        checkCudaErr(cudaMalloc((void**)&gpu_softmax_mul_target, softmax_mul_target_size));

        dim3 blockSize = dim3(SOFTMAX_MUL_BLOCK_SIZE, SOFTMAX_MUL_BLOCK_SIZE);
        dim3 gridSize = dim3((a->mat_shape->y + SOFTMAX_MUL_BLOCK_SIZE - 1) / SOFTMAX_MUL_BLOCK_SIZE, (a->mat_shape->x + SOFTMAX_MUL_BLOCK_SIZE - 1) / SOFTMAX_MUL_BLOCK_SIZE);
        softmax_mul<<<gridSize, blockSize>>>(gpu_a, gpu_b, gpu_softmax_mul_target, a->mat_shape->x, a->mat_shape->y);
        checkCudaKernelErr("softmax_mul", blockSize, gridSize);

        checkCudaErr(cudaMemcpy(&a->data[0], gpu_a, a_size, cudaMemcpyDeviceToHost));

        // SUM1 - LOG - NEG

        float* gpu_sum1_target;
        size_t sum1_target_size = a->mat_shape->x * sizeof(float);
        checkCudaErr(cudaMalloc((void**)&gpu_sum1_target, sum1_target_size));

        blockSize.x = BLOCK_SIZE;
        blockSize.y = 1;
        gridSize.x = (a->mat_shape->x + BLOCK_SIZE - 1) / BLOCK_SIZE;
        gridSize.y = 1;
        sum1_log_neg<<<gridSize, blockSize>>>(gpu_softmax_mul_target, gpu_sum1_target, a->mat_shape->y, a->mat_shape->x);
        checkCudaKernelErr("sum1_log_neg", blockSize, gridSize);

        // SUM0

        sum0<<<gridSize, blockSize>>>(gpu_sum1_target, gpu_target, 1, a->mat_shape->x);
        checkCudaKernelErr("sum0", blockSize, gridSize);

        // copy back

        checkCudaErr(cudaMemcpy(&target->data[0], gpu_target, target_size, cudaMemcpyDeviceToHost));
        target->data[0] /= a->mat_shape->x;

        cudaFree(gpu_a);
        cudaFree(gpu_b);
        cudaFree(gpu_softmax_mul_target);
        cudaFree(gpu_sum1_target);
        cudaFree(gpu_target);
    }

}

