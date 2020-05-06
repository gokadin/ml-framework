#ifndef ADD_H
#define ADD_H

#include "tensor.h"

int add_forward(TENSOR *tensor, TENSOR *a, TENSOR *b);
int add_backward(TENSOR *tensor, TENSOR *a, TENSOR *b);
int gpu_add_forward(TENSOR *tensor, TENSOR *a, TENSOR *b);
int cpu_add_forward(TENSOR *tensor, TENSOR *a, TENSOR *b);
int cpu_add_backward(TENSOR *tensor, TENSOR *a, TENSOR *b);

#endif
