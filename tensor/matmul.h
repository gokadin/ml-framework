#ifndef MATMUL_H
#define MATMUL_H

#include "tensor.h"

OP *alloc_matmul(TENSOR *a, TENSOR *b);
SHAPE matmul_target_shape(TENSOR *tensor);
int matmul_forward(TENSOR *target);
int matmul_backward(TENSOR *target);
int gpu_matmul_forward(TENSOR *a, TENSOR *b, TENSOR *target);
int cpu_matmul_forward(TENSOR *a, TENSOR *b, TENSOR *target);
int gpu_matmul_backward(TENSOR *tensor, TENSOR *a, TENSOR *b);
int cpu_matmul_backward(TENSOR *tensor, TENSOR *a, TENSOR *b);

#endif
