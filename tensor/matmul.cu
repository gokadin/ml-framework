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

__global__ void transpose(float* mat_in, float* mat_out, int rows, int cols)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows)
    {
        unsigned int pos = idy * cols + idx;
        unsigned int trans_pos = idx * rows + idy;
        mat_out[trans_pos] = mat_in[pos];
    }
}

extern "C" {

    void matmul(const TENSOR *a, const TENSOR* b, TENSOR *target)
    {
        float* gpu_a;
        size_t size = a->mat_shape.x * a->mat_shape.y * sizeof(float);
        cudaMalloc((void**)&gpu_a, size);
        cudaMemcpy(gpu_a, &a->data[0], size, cudaMemcpyHostToDevice);

        float* gpu_b;
        size = b->mat_shape.x * b->mat_shape.y * sizeof(float);
        cudaMalloc((void**)&gpu_b, size);
        cudaMemcpy(gpu_b, &b->data[0], size, cudaMemcpyHostToDevice);

        float* gpu_target;
        size = target->mat_shape.x * target->mat_shape.y * sizeof(float);
        cudaMalloc(&gpu_target, size);

        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridSize((b->mat_shape.y + BLOCK_SIZE - 1) / BLOCK_SIZE, (a->mat_shape.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
        matmul<<<gridSize, blockSize>>>(gpu_a, gpu_b, gpu_target, a->mat_shape.x, a->mat_shape.y, b->mat_shape.y);

        cudaMemcpy(&target->data[0], gpu_target, size, cudaMemcpyDeviceToHost);

        cudaFree(gpu_a);
        cudaFree(gpu_b);
        cudaFree(gpu_target);
    }

    void matmul_backward(const TENSOR *tensor, const TENSOR *a, TENSOR *b)
    {
        float* gpu_tensor_grad;
        size_t gpu_tensor_grad_size = tensor->mat_shape.x * tensor->mat_shape.y * sizeof(float);
        cudaMalloc((void**)&gpu_tensor_grad, gpu_tensor_grad_size);
        cudaMemcpy(gpu_tensor_grad, &tensor->grad[0], gpu_tensor_grad_size, cudaMemcpyHostToDevice);

        float* gpu_tensor_grad_transpose;
        size_t gpu_tensor_grad_transpose_size = tensor->mat_shape.x * tensor->mat_shape.y * sizeof(float);
        cudaMalloc((void**)&gpu_tensor_grad_transpose, gpu_tensor_grad_transpose_size);

        float* gpu_a;
        size_t a_size = a->mat_shape.x * a->mat_shape.y * sizeof(float);
        cudaMalloc((void**)&gpu_a, a_size);
        cudaMemcpy(gpu_a, &a->data[0], a_size, cudaMemcpyHostToDevice);

        float* gpu_a_grad;
        cudaMalloc(&gpu_a_grad, a_size);

        float* gpu_b;
        size_t b_size = b->mat_shape.x * b->mat_shape.y * sizeof(float);
        cudaMalloc((void**)&gpu_b, b_size);
        cudaMemcpy(gpu_b, &b->data[0], b_size, cudaMemcpyHostToDevice);

        float* gpu_b_grad;
        cudaMalloc(&gpu_b_grad, b_size);

        float* gpu_b_grad_transpose;
        cudaMalloc((void**)&gpu_b_grad_transpose, b_size);

        float* gpu_b_transpose;
        cudaMalloc((void**)&gpu_b_transpose, b_size);

        cudaStream_t streamA, streamB;
        cudaStreamCreate(&streamA);
        cudaStreamCreate(&streamB);

        // B TRANSPOSE

        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridSize((b->mat_shape.y + BLOCK_SIZE - 1) / BLOCK_SIZE, (b->mat_shape.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
        transpose<<<gridSize, blockSize, 0, streamA>>>(gpu_b, gpu_b_transpose, b->mat_shape.x, b->mat_shape.y);

        // A GRAD

        blockSize = dim3(BLOCK_SIZE, BLOCK_SIZE);
        gridSize = dim3((b->mat_shape.x + BLOCK_SIZE - 1) / BLOCK_SIZE, (tensor->mat_shape.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
        matmul<<<gridSize, blockSize, 0, streamA>>>(gpu_tensor_grad, gpu_b_transpose, gpu_a_grad, tensor->mat_shape.x, tensor->mat_shape.y, b->mat_shape.x);

        cudaMemcpy(&a->grad[0], gpu_a_grad, a_size, cudaMemcpyDeviceToHost);

        // TENSOR GRAD TRANSPOSE

        blockSize = dim3(BLOCK_SIZE, BLOCK_SIZE);
        gridSize = dim3((tensor->mat_shape.y + BLOCK_SIZE - 1) / BLOCK_SIZE, (tensor->mat_shape.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
        transpose<<<gridSize, blockSize, 0, streamB>>>(gpu_tensor_grad, gpu_tensor_grad_transpose, tensor->mat_shape.x, tensor->mat_shape.y);

        // B GRAD

        blockSize = dim3(BLOCK_SIZE, BLOCK_SIZE);
        gridSize = dim3((a->mat_shape.y + BLOCK_SIZE - 1) / BLOCK_SIZE, (tensor->mat_shape.y + BLOCK_SIZE - 1) / BLOCK_SIZE);
        matmul<<<gridSize, blockSize, 0, streamB>>>(gpu_tensor_grad_transpose, gpu_a, gpu_b_grad, tensor->mat_shape.y, tensor->mat_shape.x, a->mat_shape.y);

        // B GRAD TRANSPOSE

        blockSize = dim3(BLOCK_SIZE, BLOCK_SIZE);
        gridSize = dim3((b->mat_shape.x + BLOCK_SIZE - 1) / BLOCK_SIZE, (b->mat_shape.y + BLOCK_SIZE - 1) / BLOCK_SIZE);
        transpose<<<gridSize, blockSize, 0, streamB>>>(gpu_b_grad, gpu_b_grad_transpose, b->mat_shape.y, b->mat_shape.x);

        cudaMemcpy(&b->grad[0], gpu_b_grad_transpose, b_size, cudaMemcpyDeviceToHost);

        cudaStreamDestroy(streamA);
        cudaStreamDestroy(streamB);

        cudaFree(gpu_tensor_grad);
        cudaFree(gpu_tensor_grad_transpose);
        cudaFree(gpu_a);
        cudaFree(gpu_b);
        cudaFree(gpu_b_transpose);
        cudaFree(gpu_b_grad_transpose);
        cudaFree(gpu_a_grad);
        cudaFree(gpu_b_grad);
    }

}

