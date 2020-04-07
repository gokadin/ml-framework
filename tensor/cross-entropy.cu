#include <stdio.h>
#include <cuda.h>
#include "tensor.h"

__global__ void cross_entropy(float *a, float* b, float *target, int size_a_x, int size_a_y)
{
    extern __shared__ float r[];

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    r[index] = a[index];
    __syncthreads();

    for (int s = 0; s < size_a_y; s++) {
        int i = index / size_a_y;
        int j = i * size_a_y + s;
        target[i] += r[j];
        __syncthreads();
    }
}

extern "C" {

    void cross_entropy(TENSOR *a, TENSOR* b, TENSOR *target) {
        int size = a->shapeX * a->shapeY;
        int msize = size * sizeof(float);
        int msize_target = a->shapeX * sizeof(float);
        float* gpu_a;
        float* gpu_b;
        float* gpu_target;

        cudaMalloc((void**)&gpu_a, msize);
        cudaMemcpy(gpu_a, &a->data[0], msize, cudaMemcpyHostToDevice);
        cudaMalloc((void**)&gpu_b, msize);
        cudaMemcpy(gpu_b, &b->data[0], msize, cudaMemcpyHostToDevice);
        cudaMalloc((void**)&gpu_target, msize_target);

        dim3 blockSize = dim3(size);
        dim3 gridSize = dim3((size + blockSize.x - 1) / blockSize.x);
        cross_entropy<<<gridSize, blockSize, msize>>>(gpu_a, gpu_b, gpu_target, a->shapeX, a->shapeY);

        cudaMemcpy(&target->data[0], gpu_target, msize_target, cudaMemcpyDeviceToHost);

        cudaFree(gpu_a);
        cudaFree(gpu_b);
        cudaFree(gpu_target);
    }

}

