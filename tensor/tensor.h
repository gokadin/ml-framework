#ifndef TENSOR_H
#define TENSOR_H

#include <stdlib.h>

//typedef enum
//{
//    ADD,
//    MATMUL
//} OP_TYPE;

//typedef struct {
//    OP_TYPE op_type;
//    struct TENSOR *dependencies;
//} TENSOR_GRAPH;

typedef struct {
    int x;
    int y;
} SHAPE;

typedef struct {
    float *data;
    float *grad;
    SHAPE mat_shape;
    SHAPE grad_shape;
} TENSOR;

TENSOR *alloc_tensor();

void free_tensor(TENSOR *p);

//void set_op(OP_TYPE op_type, TENSOR *tensor, TENSOR *a, TENSOR *b);

//void forward(TENSOR *tensor);

#endif
