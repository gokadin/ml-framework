#ifndef TENSOR_H
#define TENSOR_H

#include <stdlib.h>
#include <stdarg.h>
#include <stdbool.h>
#include "shape.h"
#include "op.h"

typedef struct TENSOR {
    float *data;
    float *grad;
    SHAPE *mat_shape;
    SHAPE *grad_shape;
    bool run_on_gpu;
    int id;
    OP *op;
} TENSOR;

TENSOR *alloc_tensor(int id);
void free_tensor(TENSOR *p);

SHAPE calculate_op_shape(struct TENSOR *tensor);

int forward(TENSOR *tensor);
int backward(TENSOR *tensor);

#endif
