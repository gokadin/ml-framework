#ifndef ADD_H
#define ADD_H

#include "tensor.h"

int add_forward(TENSOR *target);
int add_backward(TENSOR *target);
int gpu_add(TENSOR *a, TENSOR *b, TENSOR *target);
int cpu_add_forward(TENSOR *a, TENSOR *b, TENSOR *target);
int cpu_add_backward(TENSOR *a, TENSOR *b, TENSOR *target);
SHAPE add_target_shape(TENSOR *tensor);

OP *alloc_add(TENSOR *a, TENSOR *b)
{
    OP *op = (OP*)malloc(sizeof(OP));

    op->forward = add_forward;
    op->backward = add_backward;
    op->target_shape = add_target_shape;

    op->operands = malloc(sizeof(TENSOR*) * 2);
    op->operands[0] = a;
    op->operands[1] = b;

    return op;
}

SHAPE add_target_shape(TENSOR *tensor)
{
    SHAPE shape;
    shape.x = tensor->op->operands[0]->mat_shape->x;
    shape.y = tensor->op->operands[0]->mat_shape->y;
    shape.size = shape.x * shape.y;
    return shape;
}

int add_forward(TENSOR *target)
{
    if (!shapes_equal(target->op->operands[0]->mat_shape, target->op->operands[1]->mat_shape))
    {
        return 1;
    }

    if (target->run_on_gpu)
    {
        return gpu_add(target->op->operands[0], target->op->operands[1], target);
    }

    return cpu_add_forward(target->op->operands[0], target->op->operands[1], target);
}

int add_backward(TENSOR *target)
{
    return cpu_add_backward(target->op->operands[0], target->op->operands[1], target);
}

int cpu_add_forward(TENSOR *a, TENSOR *b, TENSOR *target)
{
    for (int i = 0; i < target->mat_shape->size; i++)
    {
        target->data[i] = a->data[i] + b->data[i];
    }

    return 0;
}

int cpu_add_backward(TENSOR *a, TENSOR *b, TENSOR *target)
{
    for (int i = 0; i < target->grad_shape->size; i++)
    {
        a->grad[i] = target->grad[i];
        b->grad[i] = target->grad[i];
    }

    return 0;
}

#endif
