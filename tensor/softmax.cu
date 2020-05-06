#include <stdio.h>
#include <cuda.h>
#include "cudautils.h"
#include "tensor.h"

const int BLOCK_SIZE = 32;

__global__ void softmax(float *a, float *target, int x, int y)
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
            target[row * y + j] = exp(a[row * y + j]) / sum;
        }
    }
}

extern "C" {

    void gpu_softmax_forward(TENSOR *target, TENSOR *a) {
        float* gpu_a;
        size_t a_size = a->mat_shape->size * sizeof(float);
        checkCudaErr(cudaMalloc((void**)&gpu_a, a_size));
        checkCudaErr(cudaMemcpy(gpu_a, &a->data[0], a_size, cudaMemcpyHostToDevice));

        float* gpu_target;
        size_t target_size = target->mat_shape->size * sizeof(float);
        checkCudaErr(cudaMalloc((void**)&gpu_target, target_size));

        dim3 blockSize = dim3(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridSize = dim3((a->mat_shape->y + BLOCK_SIZE - 1) / BLOCK_SIZE, (a->mat_shape->x + BLOCK_SIZE - 1) / BLOCK_SIZE);
        softmax<<<gridSize, blockSize>>>(gpu_a, gpu_target, a->mat_shape->x, a->mat_shape->y);
        checkCudaKernelErr("softmax", blockSize, gridSize);

        checkCudaErr(cudaMemcpy(&target->data[0], gpu_target, target_size, cudaMemcpyDeviceToHost));

        cudaFree(gpu_a);
        cudaFree(gpu_target);
    }

}

