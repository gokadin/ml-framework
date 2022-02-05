#include <stdio.h>
#include <cuda.h>
#include "cudautils.h"
#include "tensor.h"

const int BLOCK_SIZE = 1024;

__global__ void mul(float *a, float *b, float *target, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < size; i += stride) {
        target[i] = a[i] * b[i];
    }
}

__global__ void mul_grad(float *a, float *ag, float *b, float *bg, float *cg, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < size; i += stride) {
        ag[i] = cg[i] * b[i];
        bg[i] = cg[i] * a[i];
    }
}

extern "C" {

    __declspec(dllexport) int gpu_mul_forward(TENSOR *target, TENSOR* a, TENSOR *b) {
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
        mul<<<gridSize, blockSize>>>(gpu_a, gpu_b, gpu_target, target->mat_shape->size);
        checkCudaKernelErr("add", blockSize, gridSize);

        checkCudaErr(cudaMemcpy(&target->data[0], gpu_target, target_size, cudaMemcpyDeviceToHost));

        cudaFree(gpu_a);
        cudaFree(gpu_b);
        cudaFree(gpu_target);

        return 0;
    }

    __declspec(dllexport) int gpu_mul_backward(TENSOR *target, TENSOR* a, TENSOR *b) {
        float* gpu_a;
        size_t a_size = a->mat_shape->size * sizeof(float);
        checkCudaErr(cudaMalloc((void**)&gpu_a, a_size));
        checkCudaErr(cudaMemcpy(gpu_a, &a->data[0], a_size, cudaMemcpyHostToDevice));

        float* gpu_a_grad;
        size_t a_grad_size = a->grad_shape->size * sizeof(float);
        checkCudaErr(cudaMalloc(&gpu_a_grad, a_grad_size));

        float* gpu_b;
        size_t b_size = b->mat_shape->size * sizeof(float);
        checkCudaErr(cudaMalloc((void**)&gpu_b, b_size));
        checkCudaErr(cudaMemcpy(gpu_b, &b->data[0], b_size, cudaMemcpyHostToDevice));

        float* gpu_b_grad;
        size_t b_grad_size = b->grad_shape->size * sizeof(float);
        checkCudaErr(cudaMalloc(&gpu_b_grad, b_grad_size));

        float* gpu_tensor_grad;
        size_t tensor_grad_size = target->grad_shape->size * sizeof(float);
        checkCudaErr(cudaMalloc((void**)&gpu_tensor_grad, tensor_grad_size));
        checkCudaErr(cudaMemcpy(gpu_tensor_grad, &target->grad[0], tensor_grad_size, cudaMemcpyHostToDevice));

        dim3 blockSize = dim3(BLOCK_SIZE);
        dim3 gridSize = dim3((target->mat_shape->size + BLOCK_SIZE - 1) / BLOCK_SIZE);
        mul_grad<<<gridSize, blockSize>>>(gpu_a, gpu_a_grad, gpu_b, gpu_b_grad, gpu_tensor_grad, target->grad_shape->size);
        checkCudaKernelErr("add", blockSize, gridSize);

        checkCudaErr(cudaMemcpy(&a->grad[0], gpu_a_grad, a_grad_size, cudaMemcpyDeviceToHost));
        checkCudaErr(cudaMemcpy(&b->grad[0], gpu_b_grad, b_grad_size, cudaMemcpyDeviceToHost));

        cudaFree(gpu_a);
        cudaFree(gpu_a_grad);
        cudaFree(gpu_b);
        cudaFree(gpu_b_grad);
        cudaFree(gpu_tensor_grad);

        return 0;
    }

}

