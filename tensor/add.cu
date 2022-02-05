#include <stdio.h>
#include <cuda.h>
#include "tensor.h"
#include "cudautils.h"

const int BLOCK_SIZE = 1024;

__global__ void add(float *a, float* b, float *target, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < size; i += stride) {
        target[i] = a[i] + b[i];
    }
}

extern "C" {

    __declspec(dllexport) int gpu_add_forward(TENSOR *target, TENSOR* a, TENSOR *b) {
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

        dim3 blockSize = dim3(BLOCK_SIZE);
        dim3 gridSize = dim3((target->mat_shape->size + BLOCK_SIZE - 1) / BLOCK_SIZE);
        add<<<gridSize, blockSize>>>(gpu_a, gpu_b, gpu_target, target->mat_shape->size);
        checkCudaKernelErr("add", blockSize, gridSize);

        checkCudaErr(cudaMemcpy(&target->data[0], gpu_target, target_size, cudaMemcpyDeviceToHost));

        cudaFree(gpu_a);
        cudaFree(gpu_b);
        cudaFree(gpu_target);

        return 0;
    }

}

