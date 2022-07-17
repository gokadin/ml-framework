#ifndef TENSOR_H
#define TENSOR_H

#include <stdlib.h>
#include <stdarg.h>
#include <stdbool.h>

typedef struct TENSOR {
    float *data;
    float *grad;
    int *mat_shape;
    int mat_size;
    int *grad_shape;
    int grad_size;
    bool run_on_gpu;
    int id;
} TENSOR;

TENSOR *alloc_tensor(int id);
void free_tensor(TENSOR *p);

#endif
