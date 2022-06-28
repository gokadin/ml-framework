#ifndef SOFTMAX_CROSS_ENTROPY_H
#define SOFTMAX_CROSS_ENTROPY_H

#include <math.h>
#include "tensor.h"

int sce_forward(TENSOR *tensor, TENSOR *a, TENSOR *b);
int sce_backward(TENSOR *tensor, TENSOR *a, TENSOR *b);
int gpu_sce_forward(TENSOR *tensor, TENSOR *a, TENSOR *b);
int cpu_sce_forward(TENSOR *tensor, TENSOR *a, TENSOR *b);
int gpu_sce_backward(TENSOR *tensor, TENSOR *a, TENSOR *b);
int cpu_sce_backward(TENSOR *tensor, TENSOR *a, TENSOR *b);

int sce_forward(TENSOR *target, TENSOR *a, TENSOR *b)
{
    if (target->run_on_gpu)
    {
        return gpu_sce_forward(target, a, b);
    }

    return cpu_sce_forward(target, a, b);
}

int sce_backward(TENSOR *tensor, TENSOR *a, TENSOR *b)
{
    return cpu_sce_backward(tensor, a, b);
}

int cpu_sce_forward(TENSOR *target, TENSOR *a, TENSOR *b)
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
            a->data[index] = (expf(a->data[index]) / sum);
        }
    }

    float sum1[a->mat_shape->x];
    for (int i = 0; i < a->mat_shape->x; i++)
    {
        sum1[i] = 0;
        for (int j = 0; j < a->mat_shape->y; j++)
        {
            int index = i * a->mat_shape->y + j;
            sum1[i] += b->data[index] * logf(a->data[index]);
        }
        sum1[i] = -sum1[i];
    }

    float sum0 = 0;
    for (int i = 0; i < a->mat_shape->x; i++)
    {
        sum0 += sum1[i];
    }

    target->data[0] = sum0 / a->mat_shape->x;

    return 0;
}

int cpu_sce_backward(TENSOR *tensor, TENSOR *a, TENSOR *b)
{
    for (int i = 0; i < a->mat_shape->size; i++)
    {
        a->grad[i] = tensor->grad[0] * (a->data[i] - b->data[i]);
    }

    return 0;
}

#endif
