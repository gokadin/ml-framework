#include <stdio.h>
#include <cuda.h>
#include "tensor.h"

const int BLOCK_SIZE = 1024;

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

__global__ void relu_grad(float *cg, float *ag, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < size; i += stride) {
        if (cg[i] > 0) {
            ag[i] = 1;
        } else {
            ag[i] = 0;
        }
    }
}

extern "C" {

    void gpu_relu_forward(TENSOR *a, TENSOR *target) {
        int size = a->mat_shape->x * a->mat_shape->y;
        int msize = size * sizeof(float);
        float* gpu_a;
        float* gpu_target;

        cudaMalloc((void**)&gpu_a, msize);
        cudaMemcpy(gpu_a, &a->data[0], msize, cudaMemcpyHostToDevice);
        cudaMalloc((void**)&gpu_target, msize);

        dim3 blockSize = dim3(BLOCK_SIZE);
        dim3 gridSize = dim3((size + BLOCK_SIZE - 1) / BLOCK_SIZE);
        relu<<<gridSize, blockSize>>>(gpu_a, gpu_target, size);

        cudaMemcpy(&target->data[0], gpu_target, msize, cudaMemcpyDeviceToHost);

        cudaFree(gpu_a);
        cudaFree(gpu_target);
    }

    void gpu_relu_backward(TENSOR *tensor, TENSOR *a) {
        float* gpu_tensor_grad;
        size_t gpu_tensor_grad_size = tensor->grad_shape->x * tensor->grad_shape->y * sizeof(float);
        cudaMalloc((void**)&gpu_tensor_grad, gpu_tensor_grad_size);
        cudaMemcpy(gpu_tensor_grad, &tensor->grad[0], gpu_tensor_grad_size, cudaMemcpyHostToDevice);

        float* gpu_a_grad;
        size_t a_grad_size = a->mat_shape->x * a->mat_shape->y * sizeof(float);
        cudaMalloc(&gpu_a_grad, a_grad_size);

        dim3 blockSize(BLOCK_SIZE);
        dim3 gridSize = dim3((a_grad_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
        relu_grad<<<gridSize, blockSize>>>(gpu_tensor_grad, gpu_a_grad, a_grad_size);

        cudaMemcpy(&a->grad[0], gpu_a_grad, a_grad_size, cudaMemcpyDeviceToHost);

        cudaFree(gpu_tensor_grad);
        cudaFree(gpu_a_grad);
    }

}

