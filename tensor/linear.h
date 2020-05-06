#ifndef LINEAR_H
#define LINEAR_H

#include "tensor.h"
#include "matmul.h"

int linear_forward(TENSOR *tensor, TENSOR *a, TENSOR *x, TENSOR *b);
int linear_backward(TENSOR *tensor, TENSOR *a, TENSOR *x, TENSOR *b);
int gpu_linear_forward(TENSOR *tensor, TENSOR *a, TENSOR *x, TENSOR *b);
int cpu_linear_forward(TENSOR *tensor, TENSOR *a, TENSOR *x, TENSOR *b);
int gpu_linear_backward(TENSOR *tensor, TENSOR *a, TENSOR *x, TENSOR *b);
int cpu_linear_backward(TENSOR *tensor, TENSOR *a, TENSOR *x, TENSOR *b);

int linear_forward(TENSOR *tensor, TENSOR *a, TENSOR *x, TENSOR *b)
{
    if (tensor->run_on_gpu)
    {
        return gpu_linear_forward(tensor, a, x, b);
    }

    return cpu_linear_forward(tensor, a, x, b);
}

int linear_backward(TENSOR *tensor, TENSOR *a, TENSOR *x, TENSOR *b)
{
    if (tensor->run_on_gpu)
    {
        return gpu_linear_backward(tensor, a, x, b);
    }

    return cpu_linear_backward(tensor, a, x, b);
}

int cpu_linear_forward(TENSOR *tensor, TENSOR *a, TENSOR *x, TENSOR *b)
{
    cpu_matmul_forward(tensor, a, x);
    for (int i = 0; i < tensor->mat_shape->x; i++)
    {
        for (int j = 0; j < tensor->mat_shape->y; j++)
        {
            tensor->data[i * tensor->mat_shape->y + j] += b->data[j];
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

    for (int i = 0; i < tensor->grad_shape->y; i++)
    {
        float sum = 0;
        for (int j = 0; j < tensor->grad_shape->x; j++)
        {
            sum += tensor->grad[j * tensor->grad_shape->y + i];
        }
        b->grad[i] = sum;
    }

    return 0;
}

#endif
