#ifndef RELU_H
#define RELU_H

#include "tensor.h"

int relu_forward(TENSOR *target);
int relu_backward(TENSOR *target);
int gpu_relu_forward(TENSOR *a, TENSOR *target);
int cpu_relu_forward(TENSOR *a, TENSOR *target);
int gpu_relu_backward(TENSOR *tensor, TENSOR *a);
int cpu_relu_backward(TENSOR *tensor, TENSOR *a);
SHAPE relu_target_shape(TENSOR *tensor);

OP *alloc_relu(TENSOR *a)
{
    OP *op = (OP*)malloc(sizeof(OP));

    op->forward = relu_forward;
    op->backward = relu_backward;
    op->target_shape = relu_target_shape;

    op->operands = malloc(sizeof(TENSOR*));
    op->operands[0] = a;

    return op;
}

SHAPE relu_target_shape(TENSOR *tensor)
{
    SHAPE shape;
    shape.x = tensor->op->operands[0]->mat_shape->x;
    shape.y = tensor->op->operands[0]->mat_shape->y;
    shape.size = shape.x * shape.y;
    return shape;
}

int relu_forward(TENSOR *target)
{
    if (target->run_on_gpu)
    {
        return gpu_relu_forward(target->op->operands[0], target);
    }

    return cpu_relu_forward(target->op->operands[0], target);
}

int relu_backward(TENSOR *tensor)
{
    if (tensor->run_on_gpu)
    {
        return gpu_relu_backward(tensor, tensor->op->operands[0]);
    }

    return cpu_relu_backward(tensor, tensor->op->operands[0]);
}

int cpu_relu_forward(TENSOR *a, TENSOR *target)
{
    for (int i = 0; i < target->mat_shape->size; i++)
    {
        if (a->data[i] > 0)
        {
            target->data[i] = a->data[i];
        }
        else
        {
            target->data[i] = 0;
        }
    }

    return 0;
}

int cpu_relu_backward(TENSOR *tensor, TENSOR *a)
{
    for (int i = 0; i < tensor->grad_shape->size; i++)
    {
        if (tensor->grad[i] > 0)
        {
            a->grad[i] = 1;
        }
        else
        {
            a->grad[i] = 0;
        }
    }

    return 0;
}

#endif
