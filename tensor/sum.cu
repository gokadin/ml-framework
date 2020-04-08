#include <stdio.h>
#include <cuda.h>
#include "tensor.h"

const int MAX_BLOCK_SIZE = 1024;

__global__ void sum1(float *a, float *target, int size_a_x, int size_a_y)
{
    extern __shared__ float sdata[];

    int limit = size_a_x * size_a_y;
    if (limit > MAX_BLOCK_SIZE) {
        limit = (MAX_BLOCK_SIZE / size_a_y) * size_a_y;
    }
    int index = blockIdx.x * limit + threadIdx.x;
    int localIndex = threadIdx.x;

    if (threadIdx.x >= limit) {
        return;
    }

    sdata[localIndex] = a[index];
    __syncthreads();

    int section = index / size_a_y;
    int sectionBeginIndex = section * size_a_y;
    int mid = size_a_y / 2;
    if (size_a_y % 2 != 0 && index == sectionBeginIndex) {
        sdata[localIndex] += sdata[localIndex + mid + 1];
    }
    __syncthreads();

    for (int s = mid; s > 0; s /= 2) {
        if (index - sectionBeginIndex < s) {
    printf("%d ", index - sectionBeginIndex);
            sdata[localIndex] += sdata[localIndex + s];
        }

        __syncthreads();
    }

    if (index == sectionBeginIndex) {
        target[section] = sdata[localIndex];
    }
}

extern "C" {

    void sum(TENSOR *a, int axis, TENSOR *target) {
        int size = a->shapeX * a->shapeY;
        int msize = size * sizeof(float);
        int msize_target = a->shapeX * sizeof(float);
        float* gpu_a;
        float* gpu_target;

        cudaMalloc((void**)&gpu_a, msize);
        cudaMemcpy(gpu_a, &a->data[0], msize, cudaMemcpyHostToDevice);
        cudaMalloc((void**)&gpu_target, msize_target);

        dim3 blockSize = dim3(MAX_BLOCK_SIZE);
        if (size < MAX_BLOCK_SIZE) {
           blockSize = dim3(size);
        }
        dim3 gridSize = dim3((size + blockSize.x - 1) / blockSize.x);
        sum1<<<gridSize, blockSize, blockSize.x * sizeof(float)>>>(gpu_a, gpu_target, a->shapeX, a->shapeY);

        cudaMemcpy(&target->data[0], gpu_target, msize_target, cudaMemcpyDeviceToHost);

        cudaFree(gpu_a);
        cudaFree(gpu_target);
    }

}

