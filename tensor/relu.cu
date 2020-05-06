#include <stdio.h>
#include <cuda.h>
#include "cudautils.h"
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

    void gpu_relu_forward(TENSOR *target, TENSOR *a) {
        float* gpu_a;
        size_t a_size = a->mat_shape->size * sizeof(float);
        checkCudaErr(cudaMalloc((void**)&gpu_a, a_size));
        checkCudaErr(cudaMemcpy(gpu_a, &a->data[0], a_size, cudaMemcpyHostToDevice));

        float* gpu_target;
        size_t target_size = target->mat_shape->size * sizeof(float);
        checkCudaErr(cudaMalloc((void**)&gpu_target, target_size));

        dim3 blockSize = dim3(BLOCK_SIZE);
        dim3 gridSize = dim3((a->mat_shape->size + BLOCK_SIZE - 1) / BLOCK_SIZE);
        relu<<<gridSize, blockSize>>>(gpu_a, gpu_target, a->mat_shape->size);
        checkCudaKernelErr("relu", blockSize, gridSize);

        checkCudaErr(cudaMemcpy(&target->data[0], gpu_target, target_size, cudaMemcpyDeviceToHost));

        cudaFree(gpu_a);
        cudaFree(gpu_target);
    }

    void gpu_relu_backward(TENSOR *tensor, TENSOR *a) {
        float* gpu_tensor_grad;
        size_t tensor_grad_size = tensor->grad_shape->size * sizeof(float);
        checkCudaErr(cudaMalloc((void**)&gpu_tensor_grad, tensor_grad_size));
        checkCudaErr(cudaMemcpy(gpu_tensor_grad, &tensor->grad[0], tensor_grad_size, cudaMemcpyHostToDevice));

        float* gpu_a_grad;
        size_t a_grad_size = a->grad_shape->size * sizeof(float);
        checkCudaErr(cudaMalloc(&gpu_a_grad, a_grad_size));

        dim3 blockSize(BLOCK_SIZE);
        dim3 gridSize = dim3((a->grad_shape->size + BLOCK_SIZE - 1) / BLOCK_SIZE);
        relu_grad<<<gridSize, blockSize>>>(gpu_tensor_grad, gpu_a_grad, a->grad_shape->size);
        checkCudaKernelErr("relu_grad", blockSize, gridSize);

        checkCudaErr(cudaMemcpy(&a->grad[0], gpu_a_grad, a_grad_size, cudaMemcpyDeviceToHost));

        cudaFree(gpu_tensor_grad);
        cudaFree(gpu_a_grad);
    }

}

