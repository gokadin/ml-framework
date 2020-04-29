#include <stdio.h>
#include <cuda.h>
#include "tensor.h"

__global__ void relu(float *a, float *target, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < size; i += stride) {
        if (a[i] > 0) {
            target[i] = a[i];
        } else {
            target[i] = 0;
        }
    }
}

extern "C" {

    void relu(TENSOR *a, TENSOR *target) {
        int size = a->mat_shape.x * a->mat_shape.y;
        int msize = size * sizeof(float);
        float* gpu_a;
        float* gpu_target;

        cudaMalloc((void**)&gpu_a, msize);
        cudaMemcpy(gpu_a, &a->data[0], msize, cudaMemcpyHostToDevice);
        cudaMalloc((void**)&gpu_target, msize);

        dim3 blockSize = dim3(1024);
        dim3 gridSize = dim3((size + blockSize.x - 1) / blockSize.x);
        relu<<<gridSize, blockSize>>>(gpu_a, gpu_target, size);

        cudaMemcpy(&target->data[0], gpu_target, msize, cudaMemcpyDeviceToHost);

        cudaFree(gpu_a);
        cudaFree(gpu_target);
    }

}

