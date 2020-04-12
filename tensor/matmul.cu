#include <stdio.h>
#include <cuda.h>
#include "tensor.h"

const int MAX_BLOCK_SIZE = 32;

typedef struct {
    int width;
    int height;
    int stride;
    float *data;
} Matrix;

__device__ float GetElement(const Matrix a, int row, int col)
{
    return a.data[row * a.stride + col];
}

__device__ void SetElement(Matrix a, int row, int col, float value)
{
    a.data[row * a.stride + col] = value;
}

__device__ Matrix GetSubMatrix(Matrix a, int row, int col, int blockSize)
{
    Matrix sub;
    sub.width = blockSize;
    sub.height = blockSize;
    sub.stride = a.stride;
    sub.data = &a.data[a.stride * blockSize * row + blockSize * col];
    return sub;
}
__global__ void matmul(const Matrix a, const Matrix b, Matrix target)
{
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int blockSize = blockDim.x;

    Matrix targetSub = GetSubMatrix(target, blockRow, blockCol, blockSize);

    float targetValue = 0;

    int row = threadIdx.y;
    int col = threadIdx.x;

//    if (blockIdx.x * blockSize + threadIdx.x > a.height || blockIdx.y * blockSize + threadIdx.y > b.width)
//    {
//        return;
//    }

    for (int m = 0; m < (a.width / blockSize); ++m)
    {
        Matrix aSub = GetSubMatrix(a, blockRow, m, blockSize);
        Matrix bSub = GetSubMatrix(b, m, blockCol, blockSize);

        __shared__ float as[MAX_BLOCK_SIZE][MAX_BLOCK_SIZE];
        __shared__ float bs[MAX_BLOCK_SIZE][MAX_BLOCK_SIZE];

        as[row][col] = GetElement(aSub, row, col);
        bs[row][col] = GetElement(bSub, row, col);

        __syncthreads();

        for (int e = 0; e < blockSize; ++e)
        {
            targetValue += as[row][e] * bs[e][col];
        }

        __syncthreads();
    }

    SetElement(targetSub, row, col, targetValue);
}

extern "C" {

    void matmul(const TENSOR *a, const TENSOR* b, TENSOR *target) {
        Matrix gpu_a;
        gpu_a.width = gpu_a.stride = a->shapeY;
        gpu_a.height = a->shapeX;
        size_t size = a->shapeX * a->shapeY * sizeof(float);
        cudaMalloc(&gpu_a.data, size);
        cudaMemcpy(gpu_a.data, &a->data[0], size, cudaMemcpyHostToDevice);

        Matrix gpu_b;
        gpu_b.width = gpu_b.stride = b->shapeY;
        gpu_b.height = b->shapeX;
        size = b->shapeX * b->shapeY * sizeof(float);
        cudaMalloc(&gpu_b.data, size);
        cudaMemcpy(gpu_b.data, &b->data[0], size, cudaMemcpyHostToDevice);

        Matrix gpu_target;
        gpu_target.width = gpu_target.stride = target->shapeY;
        gpu_target.height = target->shapeX;
        size = target->shapeX * target->shapeY * sizeof(float);
        cudaMalloc(&gpu_target.data, size);

        int blockSizeX = target->shapeX;
        if (blockSizeX > MAX_BLOCK_SIZE)
        {
            blockSizeX = MAX_BLOCK_SIZE;
        }
        dim3 blockSize(blockSizeX, blockSizeX);
        dim3 gridSize(b->shapeY / blockSize.x, a->shapeX / blockSize.y);
        matmul<<<gridSize, blockSize>>>(gpu_a, gpu_b, gpu_target);

        cudaMemcpy(&target->data[0], gpu_target.data, size, cudaMemcpyDeviceToHost);

        cudaFree(gpu_a.data);
        cudaFree(gpu_b.data);
        cudaFree(gpu_target.data);
    }

}

