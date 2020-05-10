#ifndef MATMUL_H
#define MATMUL_H

#include "tensor.h"

int matmul_forward(TENSOR *target, TENSOR *a, TENSOR *b);
int matmul_backward(TENSOR *target, TENSOR *a, TENSOR *b);
int gpu_matmul_forward(TENSOR *tensor, TENSOR *a, TENSOR *b);
int cpu_matmul_forward(TENSOR *tensor, TENSOR *a, TENSOR *b);
int gpu_matmul_backward(TENSOR *tensor, TENSOR *a, TENSOR *b);
int cpu_matmul_backward(TENSOR *tensor, TENSOR *a, TENSOR *b);

#endif
