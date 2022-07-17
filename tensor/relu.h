#ifndef RELU_H
#define RELU_H

#include "tensor.h"

int relu_forward(TENSOR *target, TENSOR *a);
int relu_backward(TENSOR *target, TENSOR *a);
int gpu_relu_forward(TENSOR *tensor, TENSOR *a);
int cpu_relu_forward(TENSOR *tensor, TENSOR *a);
int gpu_relu_backward(TENSOR *tensor, TENSOR *a);
int cpu_relu_backward(TENSOR *tensor, TENSOR *a);

int relu_forward(TENSOR *target, TENSOR *a)
{
    if (target->run_on_gpu)
    {
        return gpu_relu_forward(target, a);
    }

    return cpu_relu_forward(target, a);
}

int relu_backward(TENSOR *tensor, TENSOR *a)
{
    if (tensor->run_on_gpu)
    {
        return gpu_relu_backward(tensor, a);
    }

    return cpu_relu_backward(tensor, a);
}

int cpu_relu_forward(TENSOR *target, TENSOR *a)
{
    for (int i = 0; i < target->mat_size; i++)
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
    for (int i = 0; i < tensor->grad_size; i++)
    {
        if (tensor->data[i] > 0)
        {
            a->grad[i] = tensor->grad[i];
        }
        else
        {
            a->grad[i] = 0;
        }
    }

    return 0;
}

#endif
