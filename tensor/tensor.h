#ifndef TENSOR_H
#define TENSOR_H

#include <stdlib.h>
#include <stdarg.h>
#include <stdbool.h>
#include "shape.h"

typedef struct TENSOR {
    float *data;
    float *grad;
    SHAPE *mat_shape;
    SHAPE *grad_shape;
    bool run_on_gpu;
    int id;
} TENSOR;

TENSOR *alloc_tensor(int id);
void free_tensor(TENSOR *p);

#endif
