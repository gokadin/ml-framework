#ifndef ADD_H
#define ADD_H

#include "tensor.h"

OP *alloc_add(TENSOR *a, TENSOR *b);
SHAPE add_target_shape(TENSOR *tensor);
int add_forward(TENSOR *target);
int add_backward(TENSOR *target);
int gpu_add_forward(TENSOR *a, TENSOR *b, TENSOR *target);
int cpu_add_forward(TENSOR *a, TENSOR *b, TENSOR *target);
int cpu_add_backward(TENSOR *tensor, TENSOR *a, TENSOR *b);

#endif
