#include <stdio.h>
#include <cuda.h>
#include "tensor.h"

__global__ void add(float *a, float* b, float *target, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < size; i += stride) {
        target[i] = a[i] + b[i];
    }
}

extern "C" {

    void add(TENSOR *a, TENSOR* b, TENSOR *target) {
        int size = a->mat_shape.x * a->mat_shape.y;
        int msize = size * sizeof(float);
        float* gpu_a;
        float* gpu_b;
        float* gpu_target;

        cudaMalloc((void**)&gpu_a, msize);
        cudaMemcpy(gpu_a, &a->data[0], msize, cudaMemcpyHostToDevice);
        cudaMalloc((void**)&gpu_b, msize);
        cudaMemcpy(gpu_b, &b->data[0], msize, cudaMemcpyHostToDevice);
        cudaMalloc((void**)&gpu_target, msize);

        dim3 blockSize = dim3(1024);
        dim3 gridSize = dim3((size + blockSize.x - 1) / blockSize.x);
        add<<<gridSize, blockSize>>>(gpu_a, gpu_b, gpu_target, size);

        cudaMemcpy(&target->data[0], gpu_target, msize, cudaMemcpyDeviceToHost);

        cudaFree(gpu_a);
        cudaFree(gpu_b);
        cudaFree(gpu_target);
    }

}

