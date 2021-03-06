#include <stdio.h>
#include <cuda.h>
#include "tensor.h"

__global__ void expand_0(float *a, float *target, int copies, int size_a_y)
{
    int indexI = blockIdx.x * blockDim.x + threadIdx.x;
    int indexJ = blockIdx.y * blockDim.y + threadIdx.y;
    int strideI = blockDim.x * gridDim.x;
    int strideJ = blockDim.y * gridDim.y;

    for (int i = indexI; i < copies; i += strideI) {
        for (int j = indexJ; j < size_a_y; j += strideJ) {
            target[i * size_a_y + j] = a[j];
        }
    }
}

__global__ void expand_1(float *a, float *target, int copies, int size_a_x)
{
    int indexI = blockIdx.x * blockDim.x + threadIdx.x;
    int indexJ = blockIdx.y * blockDim.y + threadIdx.y;
    int strideI = blockDim.x * gridDim.x;
    int strideJ = blockDim.y * gridDim.y;

    for (int i = indexI; i < size_a_x; i += strideI) {
        for (int j = indexJ; j < copies; j += strideJ) {
            target[i * copies + j] = a[i];
        }
    }
}

extern "C" {

    int gpu_expand_forward(TENSOR *target, TENSOR *a, int axis, int copies) {
        int size_a = a->mat_shape->x * a->mat_shape->y;
        int size_target = 0;
        if (axis == 0) {
            size_target = copies * a->mat_shape->y;
        } else if (axis == 1) {
            size_target = a->mat_shape->x * copies;
        }
        int msize_a = size_a * sizeof(float);
        int msize_target = size_target * sizeof(float);
        float* gpu_a;
        float* gpu_target;

        cudaMalloc((void**)&gpu_a, msize_a);
        cudaMemcpy(gpu_a, &a->data[0], msize_a, cudaMemcpyHostToDevice);
        cudaMalloc((void**)&gpu_target, msize_target);

        if (axis == 0) {
            dim3 blockSize = dim3(32, 32);
            dim3 gridSize = dim3((copies + blockSize.x - 1) / blockSize.x, (a->mat_shape->y + blockSize.y - 1) / blockSize.y);
            expand_0<<<gridSize, blockSize>>>(gpu_a, gpu_target, copies, a->mat_shape->y);
        } else {
            dim3 blockSize = dim3(32, 32);
            dim3 gridSize = dim3((copies + blockSize.x - 1) / blockSize.x, (a->mat_shape->x + blockSize.y - 1) / blockSize.y);
            expand_1<<<gridSize, blockSize>>>(gpu_a, gpu_target, copies, a->mat_shape->x);
        }

        cudaMemcpy(&target->data[0], gpu_target, msize_target, cudaMemcpyDeviceToHost);

        cudaFree(gpu_a);
        cudaFree(gpu_target);

        return 0;
    }

}

