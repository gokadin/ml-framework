#include <stdio.h>
#include <cuda.h>
#include "tensor.h"
#include "matmul.cuh"
#include "sum.cuh"

const int BLOCK_SIZE = 32;
const int BLOCK_SIZE_SUM = 1024;

__global__ void linear(float *a, float *x, float *b, float *c, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if (col < k && row < m)
    {
        for (int i = 0; i < n; i++)
        {
            sum += a[row * n + i] * x[i * k + col];
        }
        c[row * k + col] = sum + b[col];
    }
}

extern "C" {

    void gpu_linear_forward(const TENSOR *a, const TENSOR *x, const TENSOR* b, TENSOR *target)
    {
        float* gpu_a;
        size_t size = a->mat_shape->x * a->mat_shape->y * sizeof(float);
        cudaMalloc((void**)&gpu_a, size);
        cudaMemcpy(gpu_a, &a->data[0], size, cudaMemcpyHostToDevice);

        float* gpu_b;
        size_t size_b = b->mat_shape->y * sizeof(float);
        cudaMalloc((void**)&gpu_b, size_b);
        cudaMemcpy(gpu_b, &b->data[0], size_b, cudaMemcpyHostToDevice);

        float* gpu_x;
        size = x->mat_shape->x * x->mat_shape->y * sizeof(float);
        cudaMalloc((void**)&gpu_x, size);
        cudaMemcpy(gpu_x, &x->data[0], size, cudaMemcpyHostToDevice);

        float* gpu_target;
        size = target->mat_shape->x * target->mat_shape->y * sizeof(float);
        cudaMalloc(&gpu_target, size);

        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridSize((x->mat_shape->y + BLOCK_SIZE - 1) / BLOCK_SIZE, (a->mat_shape->x + BLOCK_SIZE - 1) / BLOCK_SIZE);
        linear<<<gridSize, blockSize>>>(gpu_a, gpu_x, gpu_b, gpu_target, a->mat_shape->x, a->mat_shape->y, x->mat_shape->y);

        cudaMemcpy(&target->data[0], gpu_target, size, cudaMemcpyDeviceToHost);

        cudaFree(gpu_a);
        cudaFree(gpu_x);
        cudaFree(gpu_b);
        cudaFree(gpu_target);
    }

    void gpu_linear_backward(const TENSOR *tensor, const TENSOR *a, const TENSOR *x, TENSOR *b)
    {
        float* gpu_tensor_grad;
        size_t gpu_tensor_grad_size = tensor->grad_shape->x * tensor->grad_shape->y * sizeof(float);
        cudaMalloc((void**)&gpu_tensor_grad, gpu_tensor_grad_size);
        cudaMemcpy(gpu_tensor_grad, &tensor->grad[0], gpu_tensor_grad_size, cudaMemcpyHostToDevice);

        float* gpu_a;
        size_t a_size = a->mat_shape->x * a->mat_shape->y * sizeof(float);
        cudaMalloc((void**)&gpu_a, a_size);
        cudaMemcpy(gpu_a, &a->data[0], a_size, cudaMemcpyHostToDevice);

        float* gpu_a_grad;
        cudaMalloc(&gpu_a_grad, a_size);

        float* gpu_b;
        size_t b_size = b->mat_shape->x * b->mat_shape->y * sizeof(float);
        size_t b_grad_size = a->mat_shape->x * b->mat_shape->y * sizeof(float);
        cudaMalloc((void**)&gpu_b, b_size);
        cudaMemcpy(gpu_b, &b->data[0], b_size, cudaMemcpyHostToDevice);

        float* gpu_b_grad;
        cudaMalloc(&gpu_b_grad, b_size);

        float* gpu_x;
        size_t x_size = x->mat_shape->x * x->mat_shape->y * sizeof(float);
        cudaMalloc((void**)&gpu_x, x_size);
        cudaMemcpy(gpu_x, &x->data[0], x_size, cudaMemcpyHostToDevice);

        float* gpu_x_grad;
        cudaMalloc(&gpu_x_grad, x_size);

        cudaStream_t streamA, streamB, streamC;
        cudaStreamCreate(&streamA);
        cudaStreamCreate(&streamB);
        cudaStreamCreate(&streamC);

        // A GRAD

        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridSize((x->mat_shape->x + BLOCK_SIZE - 1) / BLOCK_SIZE, (tensor->grad_shape->x + BLOCK_SIZE - 1) / BLOCK_SIZE);
        matmul_a_grad<<<gridSize, blockSize, 0, streamA>>>(gpu_tensor_grad, gpu_x, gpu_a_grad, tensor->grad_shape->x, tensor->grad_shape->y, x->mat_shape->x);

        cudaMemcpy(&a->grad[0], gpu_a_grad, a_size, cudaMemcpyDeviceToHost);

        // X GRAD

        blockSize = dim3(BLOCK_SIZE, BLOCK_SIZE);
        gridSize = dim3((a->mat_shape->y + BLOCK_SIZE - 1) / BLOCK_SIZE, (tensor->grad_shape->y + BLOCK_SIZE - 1) / BLOCK_SIZE);
        matmul_b_grad<<<gridSize, blockSize, 0, streamB>>>(gpu_tensor_grad, gpu_a, gpu_x_grad, tensor->grad_shape->y, tensor->grad_shape->x, a->mat_shape->y);

        cudaMemcpy(&x->grad[0], gpu_x_grad, x_size, cudaMemcpyDeviceToHost);

        // B GRAD

        blockSize = dim3(BLOCK_SIZE_SUM);
        gridSize = dim3((b->mat_shape->y + BLOCK_SIZE_SUM - 1) / BLOCK_SIZE_SUM);
        sum0<<<gridSize, blockSize>>>(gpu_b, gpu_b_grad, b->mat_shape->y, b->mat_shape->x);

        cudaMemcpy(&b->grad[0], gpu_b_grad, b_grad_size, cudaMemcpyDeviceToHost);

        cudaStreamDestroy(streamA);
        cudaStreamDestroy(streamB);
        cudaStreamDestroy(streamC);

        cudaFree(gpu_tensor_grad);
        cudaFree(gpu_a);
        cudaFree(gpu_x);
        cudaFree(gpu_b);
        cudaFree(gpu_a_grad);
        cudaFree(gpu_x_grad);
        cudaFree(gpu_b_grad);
    }

}

