#include <stdio.h>
#include <cuda.h>
#include "tensor.h"

const int BLOCK_SIZE = 32;

__global__ void matmul(float *a, float *b, float *c, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if (col < k && row < m)
    {
        for (int i = 0; i < n; i++)
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}

__global__ void transpose(float* a, float* c, int rows, int cols)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < cols && row < rows)
    {
        unsigned int pos = row * cols + col;
        unsigned int trans_pos = col * rows + row;
        c[trans_pos] = a[pos];
    }
}

__global__ void matmul_a_grad(float *cg, float *b, float *ag, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if (col < k && row < m)
    {
        for (int i = 0; i < n; i++)
        {
            sum += cg[row * n + i] * b[col * n + i];
        }
        ag[row * k + col] = sum;
    }
}

__global__ void matmul_b_grad(float *cg, float *a, float *bg, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if (col < k && row < m)
    {
        for (int i = 0; i < n; i++)
        {
            sum += cg[i * m + row] * a[i * k + col];
        }
        bg[col * m + row] = sum;
    }
}

extern "C" {

    void gpu_matmul_forward(const TENSOR *a, const TENSOR* b, TENSOR *target)
    {
        float* gpu_a;
        size_t size = a->mat_shape->x * a->mat_shape->y * sizeof(float);
        cudaMalloc((void**)&gpu_a, size);
        cudaMemcpy(gpu_a, &a->data[0], size, cudaMemcpyHostToDevice);

        float* gpu_b;
        size = b->mat_shape->x * b->mat_shape->y * sizeof(float);
        cudaMalloc((void**)&gpu_b, size);
        cudaMemcpy(gpu_b, &b->data[0], size, cudaMemcpyHostToDevice);

        float* gpu_target;
        size = target->mat_shape->x * target->mat_shape->y * sizeof(float);
        cudaMalloc(&gpu_target, size);

        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridSize((b->mat_shape->y + BLOCK_SIZE - 1) / BLOCK_SIZE, (a->mat_shape->x + BLOCK_SIZE - 1) / BLOCK_SIZE);
        matmul<<<gridSize, blockSize>>>(gpu_a, gpu_b, gpu_target, a->mat_shape->x, a->mat_shape->y, b->mat_shape->y);

        cudaMemcpy(&target->data[0], gpu_target, size, cudaMemcpyDeviceToHost);

        cudaFree(gpu_a);
        cudaFree(gpu_b);
        cudaFree(gpu_target);
    }

    void gpu_matmul_backward(const TENSOR *tensor, const TENSOR *a, TENSOR *b)
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
        cudaMalloc((void**)&gpu_b, b_size);
        cudaMemcpy(gpu_b, &b->data[0], b_size, cudaMemcpyHostToDevice);

        float* gpu_b_grad;
        cudaMalloc(&gpu_b_grad, b_size);

        cudaStream_t streamA, streamB;
        cudaStreamCreate(&streamA);
        cudaStreamCreate(&streamB);

        // A GRAD

        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridSize((b->mat_shape->x + BLOCK_SIZE - 1) / BLOCK_SIZE, (tensor->grad_shape->x + BLOCK_SIZE - 1) / BLOCK_SIZE);
        matmul_a_grad<<<gridSize, blockSize, 0, streamA>>>(gpu_tensor_grad, gpu_b, gpu_a_grad, tensor->grad_shape->x, tensor->grad_shape->y, b->mat_shape->x);

        cudaMemcpy(&a->grad[0], gpu_a_grad, a_size, cudaMemcpyDeviceToHost);

        // B GRAD

        blockSize = dim3(BLOCK_SIZE, BLOCK_SIZE);
        gridSize = dim3((a->mat_shape->y + BLOCK_SIZE - 1) / BLOCK_SIZE, (tensor->grad_shape->y + BLOCK_SIZE - 1) / BLOCK_SIZE);
        matmul_b_grad<<<gridSize, blockSize, 0, streamB>>>(gpu_tensor_grad, gpu_a, gpu_b_grad, tensor->grad_shape->y, tensor->grad_shape->x, a->mat_shape->y);

        cudaMemcpy(&b->grad[0], gpu_b_grad, b_size, cudaMemcpyDeviceToHost);

        cudaStreamDestroy(streamA);
        cudaStreamDestroy(streamB);

        cudaFree(gpu_tensor_grad);
        cudaFree(gpu_a);
        cudaFree(gpu_b);
        cudaFree(gpu_a_grad);
        cudaFree(gpu_b_grad);
    }

}

