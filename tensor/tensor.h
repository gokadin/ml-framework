#ifndef TENSOR_H
#define TENSOR_H

#include <stdlib.h>

typedef struct {
    float *data;
    float *grad;
    int shapeX;
    int shapeY;
} TENSOR;

TENSOR *alloc_tensor();

void free_tensor(TENSOR *p);

#endif
