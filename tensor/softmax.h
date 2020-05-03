#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <math.h>
#include "tensor.h"

OP *alloc_softmax(TENSOR *a);
SHAPE softmax_target_shape(TENSOR *tensor);
int softmax_forward(TENSOR *target);
int softmax_backward(TENSOR *tensor);
int gpu_softmax_forward(TENSOR *a, TENSOR *target);
int cpu_softmax_forward(TENSOR *a, TENSOR *target);

OP *alloc_softmax(TENSOR *a)
{
    OP *op = (OP*)malloc(sizeof(OP));

    op->forward = softmax_forward;
    op->backward = softmax_backward;
    op->target_shape = softmax_target_shape;

    op->operands = malloc(sizeof(TENSOR*));
    op->operands[0] = a;

    return op;
}

SHAPE softmax_target_shape(TENSOR *tensor)
{
    SHAPE shape;
    shape.x = tensor->mat_shape->x;
    shape.y = tensor->mat_shape->y;
    shape.size = tensor->mat_shape->size;
    return shape;
}

// TODO enable GPU forward
int softmax_forward(TENSOR *target)
{
    if (target->run_on_gpu)
    {
        return cpu_softmax_forward(target->op->operands[0], target);
//        return gpu_softmax_forward(target->op->operands[0], target);
    }

    return cpu_softmax_forward(target->op->operands[0], target);
}

int softmax_backward(TENSOR *tensor)
{
    // backward softmax is disabled
    return 0;
}

int cpu_softmax_forward(TENSOR *a, TENSOR *target)
{
    for (int i = 0; i < a->mat_shape->x; i++)
    {
        float sum = 0;
        for (int j = 0; j < a->mat_shape->y; j++)
        {
            sum += expf(a->data[i * a->mat_shape->y + j]);
        }
        for (int j = 0; j < a->mat_shape->y; j++)
        {
            int index = i * a->mat_shape->y + j;
            target->data[index] = expf(a->data[index]) / sum;
        }
    }

    return 0;
}

#endif
