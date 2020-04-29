#ifndef ADD_H
#define ADD_H

#include "tensor.h"

int add_forward(TENSOR *target);
int add_backward(TENSOR *target);
int gpu_add(TENSOR *a, TENSOR *b, TENSOR *target);
int cpu_add_forward(TENSOR *a, TENSOR *b, TENSOR *target);
int cpu_add_backward(TENSOR *a, TENSOR *b, TENSOR *target);

typedef struct {
    OP op;
} OP_ADD;

OP *alloc_add(TENSOR *a, TENSOR *b)
{
    OP *op = (OP*)malloc(sizeof(OP_ADD));
    op->forward = add_forward;
    op->backward = add_backward;
    op->dependencies = malloc(sizeof(TENSOR) * 2);
    op->dependencies[0] = *a;
    op->dependencies[1] = *b;
    return op;
}

int add_forward(TENSOR *target)
{
    if (target->run_on_gpu)
    {
        return gpu_add(&target->op->dependencies[0], &target->op->dependencies[1], target);
    }

    return cpu_add_forward(&target->op->dependencies[0], &target->op->dependencies[1], target);
}

int add_backward(TENSOR *target)
{
    return cpu_add_backward(&target->op->dependencies[0], &target->op->dependencies[1], target);
}

int cpu_add_forward(TENSOR *a, TENSOR *b, TENSOR *target)
{
    for (int i = 0; i < target->mat_shape.size; i++)
    {
        target->data[i] = a->data[i] + b->data[i];
    }

    return 0;
}

int cpu_add_backward(TENSOR *a, TENSOR *b, TENSOR *target)
{
    for (int i = 0; i < target->grad_shape.size; i++)
    {
        a->grad[i] = target->grad[i];
        b->grad[i] = target->grad[i];
    }

    return 0;
}

#endif
