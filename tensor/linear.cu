#include <stdio.h>
#include <cuda.h>
#include "cudautils.h"
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

    __declspec(dllexport) int gpu_linear_forward(const TENSOR *target, const TENSOR *a, const TENSOR* x, TENSOR *b)
    {
        float* gpu_a;
        size_t a_size = a->mat_size * sizeof(float);
        checkCudaErr(cudaMalloc((void**)&gpu_a, a_size));
        checkCudaErr(cudaMemcpy(gpu_a, &a->data[0], a_size, cudaMemcpyHostToDevice));

        float* gpu_b;
        size_t b_size = b->mat_size * sizeof(float);
        checkCudaErr(cudaMalloc((void**)&gpu_b, b_size));
        checkCudaErr(cudaMemcpy(gpu_b, &b->data[0], b_size, cudaMemcpyHostToDevice));

        float* gpu_x;
        size_t x_size = x->mat_size * sizeof(float);
        checkCudaErr(cudaMalloc((void**)&gpu_x, x_size));
        checkCudaErr(cudaMemcpy(gpu_x, &x->data[0], x_size, cudaMemcpyHostToDevice));

        float* gpu_target;
        size_t target_size = target->mat_size * sizeof(float);
        checkCudaErr(cudaMalloc(&gpu_target, target_size));

        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridSize((x->mat_shape[1] + BLOCK_SIZE - 1) / BLOCK_SIZE, (a->mat_shape[0] + BLOCK_SIZE - 1) / BLOCK_SIZE);
        linear<<<gridSize, blockSize>>>(gpu_a, gpu_x, gpu_b, gpu_target, a->mat_shape[0], a->mat_shape[1], x->mat_shape[1]);
        checkCudaKernelErr("linear", blockSize, gridSize);

        checkCudaErr(cudaMemcpy(&target->data[0], gpu_target, target_size, cudaMemcpyDeviceToHost));

        cudaFree(gpu_a);
        cudaFree(gpu_x);
        cudaFree(gpu_b);
        cudaFree(gpu_target);

        return 0;
    }

    __declspec(dllexport) int gpu_linear_backward(const TENSOR *tensor, const TENSOR *a, const TENSOR *x, TENSOR *b)
    {
        float* gpu_tensor_grad;
        size_t gpu_tensor_grad_size = tensor->grad_size * sizeof(float);
        checkCudaErr(cudaMalloc((void**)&gpu_tensor_grad, gpu_tensor_grad_size));
        checkCudaErr(cudaMemcpy(gpu_tensor_grad, &tensor->grad[0], gpu_tensor_grad_size, cudaMemcpyHostToDevice));

        float* gpu_a;
        size_t a_size = a->mat_size * sizeof(float);
        checkCudaErr(cudaMalloc((void**)&gpu_a, a_size));
        checkCudaErr(cudaMemcpy(gpu_a, &a->data[0], a_size, cudaMemcpyHostToDevice));

        float* gpu_a_grad;
        checkCudaErr(cudaMalloc(&gpu_a_grad, a_size));

        float* gpu_b;
        size_t b_size = b->mat_size * sizeof(float);
        checkCudaErr(cudaMalloc((void**)&gpu_b, b_size));
        checkCudaErr(cudaMemcpy(gpu_b, &b->data[0], b_size, cudaMemcpyHostToDevice));

        float* gpu_b_grad;
        size_t b_grad_size = b->grad_size * sizeof(float);
        checkCudaErr(cudaMalloc(&gpu_b_grad, b_grad_size));

        float* gpu_x;
        size_t x_size = x->mat_size * sizeof(float);
        checkCudaErr(cudaMalloc((void**)&gpu_x, x_size));
        checkCudaErr(cudaMemcpy(gpu_x, &x->data[0], x_size, cudaMemcpyHostToDevice));

        float* gpu_x_grad;
        checkCudaErr(cudaMalloc(&gpu_x_grad, x_size));

        cudaStream_t streamA, streamB;
        cudaStreamCreate(&streamA);
        cudaStreamCreate(&streamB);

        // A GRAD

        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridSize((x->mat_shape[0] + BLOCK_SIZE - 1) / BLOCK_SIZE, (tensor->grad_shape[0] + BLOCK_SIZE - 1) / BLOCK_SIZE);
        matmul_a_grad<<<gridSize, blockSize, 0, streamA>>>(gpu_tensor_grad, gpu_x, gpu_a_grad, tensor->grad_shape[0], tensor->grad_shape[1], x->mat_shape[0]);
        checkCudaKernelErr("matmul_a_grad", blockSize, gridSize);

        checkCudaErr(cudaMemcpy(&a->grad[0], gpu_a_grad, a_size, cudaMemcpyDeviceToHost));

        // X GRAD

        gridSize.x = (a->mat_shape[1] + BLOCK_SIZE - 1) / BLOCK_SIZE;
        gridSize.y = (tensor->grad_shape[1] + BLOCK_SIZE - 1) / BLOCK_SIZE;
        matmul_b_grad<<<gridSize, blockSize, 0, streamB>>>(gpu_tensor_grad, gpu_a, gpu_x_grad, tensor->grad_shape[1], tensor->grad_shape[0], a->mat_shape[1]);
        checkCudaKernelErr("matmul_b_grad", blockSize, gridSize);

        checkCudaErr(cudaMemcpy(&x->grad[0], gpu_x_grad, x_size, cudaMemcpyDeviceToHost));

        // B GRAD

        blockSize.x = BLOCK_SIZE_SUM;
        blockSize.y = 1;
        gridSize.x = (tensor->grad_shape[1] + BLOCK_SIZE_SUM - 1) / BLOCK_SIZE_SUM;
        gridSize.y = 1;
        sum0<<<gridSize, blockSize>>>(gpu_tensor_grad, gpu_b_grad, tensor->grad_shape[1], tensor->grad_shape[0]);
        checkCudaKernelErr("sum0", blockSize, gridSize);

        checkCudaErr(cudaMemcpy(&b->grad[0], gpu_b_grad, b_grad_size, cudaMemcpyDeviceToHost));

        cudaStreamDestroy(streamA);
        cudaStreamDestroy(streamB);

        cudaFree(gpu_tensor_grad);
        cudaFree(gpu_a);
        cudaFree(gpu_x);
        cudaFree(gpu_b);
        cudaFree(gpu_a_grad);
        cudaFree(gpu_x_grad);
        cudaFree(gpu_b_grad);

        return 0;
    }

}

