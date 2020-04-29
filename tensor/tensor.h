#ifndef TENSOR_H
#define TENSOR_H

#include <stdlib.h>
#include <stdarg.h>
#include <stdbool.h>
#include "op.h"

struct TENSOR;

typedef struct {
    int x;
    int y;
    int size;
} SHAPE;

typedef struct TENSOR {
    float *data;
    float *grad;
    SHAPE mat_shape;
    SHAPE grad_shape;
    bool run_on_gpu;
    OP *op;
} TENSOR;

TENSOR *alloc_tensor();

void free_tensor(TENSOR *p);

void set_mat_shape(TENSOR *tensor, int x, int y);

void set_grad_shape(TENSOR *tensor, int x, int y);

int forward(TENSOR *tensor);

int backward(TENSOR *tensor);

#endif
