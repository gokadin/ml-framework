#ifndef LINEAR_H
#define LINEAR_H

#include "tensor.h"
#include "matmul.h"
#include "add.h"

int linear_forward(TENSOR *target);
int linear_backward(TENSOR *target);
int gpu_linear_forward(TENSOR *a, TENSOR *x, TENSOR *b, TENSOR *target);
int cpu_linear_forward(TENSOR *a, TENSOR *x, TENSOR *b, TENSOR *target);
int gpu_linear_backward(TENSOR *tensor, TENSOR *a, TENSOR *x, TENSOR *b);
int cpu_linear_backward(TENSOR *tensor, TENSOR *a, TENSOR *x, TENSOR *b);
SHAPE linear_target_shape(TENSOR *tensor);

OP *alloc_linear(TENSOR *a, TENSOR *x, TENSOR *b)
{
    OP *op = (OP*)malloc(sizeof(OP));

    op->forward = linear_forward;
    op->backward = linear_backward;
    op->target_shape = linear_target_shape;

    op->operands = malloc(sizeof(TENSOR*) * 3);
    op->operands[0] = a;
    op->operands[1] = x;
    op->operands[2] = b;

    return op;
}

SHAPE linear_target_shape(TENSOR *tensor)
{
    SHAPE shape;
    shape.x = tensor->op->operands[0]->mat_shape->x;
    shape.y = tensor->op->operands[1]->mat_shape->y;
    shape.size = shape.x * shape.y;
    return shape;
}

int linear_forward(TENSOR *target)
{
    if (target->op->operands[0]->mat_shape->y != target->op->operands[1]->mat_shape->x || target->op->operands[2]->mat_shape->x != 1)
    {
        return 1;
    }

    if (target->run_on_gpu)
    {
        return gpu_linear_forward(target->op->operands[0], target->op->operands[1], target->op->operands[2], target);
    }

    return cpu_linear_forward(target->op->operands[0], target->op->operands[1], target->op->operands[2], target);
}

int linear_backward(TENSOR *target)
{
    if (target->run_on_gpu)
    {
        return gpu_linear_backward(target, target->op->operands[0], target->op->operands[1], target->op->operands[2]);
    }

    return cpu_linear_backward(target, target->op->operands[0], target->op->operands[1], target->op->operands[2]);
}

int cpu_linear_forward(TENSOR *a, TENSOR *x, TENSOR *b, TENSOR *target)
{
    cpu_matmul_forward(a, x, target);
    for (int i = 0; i < target->mat_shape->x; i++)
    {
        for (int j = 0; j < target->mat_shape->y; j++)
        {
            target->data[i * target->mat_shape->y + j] += b->data[j];
        }
    }

    return 0;
}

int cpu_linear_backward(TENSOR *tensor, TENSOR *a, TENSOR *x, TENSOR *b)
{
    int code = cpu_matmul_backward(tensor, a, x);
    if (code != 0)
    {
        return code;
    }

    for (int i = 0; i < b->grad_shape->y; i++)
    {
        float sum = 0;
        for (int j = 0; j < tensor->grad_shape->x; j++)
        {
            sum += tensor->grad[i * b->grad_shape->y + j];
        }
        b->grad[i] = sum;
    }

    return 0;
}

#endif
