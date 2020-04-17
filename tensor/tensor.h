#ifndef TENSOR_H
#define TENSOR_H

#include <stdlib.h>

//typedef enum
//{
//    ADD,
//    MATMUL
//} OP_TYPE;

struct TENSOR;

/* TENSOR_GRAPH */

//typedef struct {
//    OP_TYPE op_type;
//    struct TENSOR *dependencies;
//} TENSOR_GRAPH;

/* TENSOR */

typedef struct TENSOR {
    float *data;
    float *grad;
    int shapeX;
    int shapeY;
//    TENSOR_GRAPH graph;
} TENSOR;

TENSOR *alloc_tensor();

void free_tensor(TENSOR *p);

//void set_op(OP_TYPE op_type, TENSOR *tensor, TENSOR *a, TENSOR *b);

//void forward(TENSOR *tensor);

#endif
