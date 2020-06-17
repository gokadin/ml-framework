#ifndef MUL_H
#define MUL_H

#include "tensor.h"

int mul_forward(TENSOR *target, TENSOR *a, TENSOR *b);
int mul_backward(TENSOR *target, TENSOR *a, TENSOR *b);
int gpu_mul_forward(TENSOR *tensor, TENSOR *a, TENSOR *b);
int cpu_mul_forward(TENSOR *tensor, TENSOR *a, TENSOR *b);
int gpu_mul_backward(TENSOR *tensor, TENSOR *a, TENSOR *b);
int cpu_mul_backward(TENSOR *tensor, TENSOR *a, TENSOR *b);

int mul_forward(TENSOR *target, TENSOR *a, TENSOR *b)
{
    if (target->run_on_gpu)
    {
        return gpu_mul_forward(target, a, b);
    }

    return cpu_mul_forward(target, a, b);
}

int mul_backward(TENSOR *tensor, TENSOR *a, TENSOR *b)
{
    if (tensor->run_on_gpu)
    {
        return gpu_mul_backward(tensor, a, b);
    }

    return cpu_mul_backward(tensor, a, b);
}

int cpu_mul_forward(TENSOR *tensor, TENSOR *a, TENSOR *b)
{
    for (int i = 0; i < tensor->mat_shape->size; i++)
    {
        tensor->data[i] = a->data[i] * b->data[i];
    }

    return 0;
}

int cpu_mul_backward(TENSOR *tensor, TENSOR *a, TENSOR *b)
{
    for (int i = 0; i < tensor->grad_shape->size; i++)
    {
        a->grad[i] = tensor->grad[i] * b->data[i];
        b->grad[i] = tensor->grad[i] * a->data[i];
    }

    return 0;
}

#endif
