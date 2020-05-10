#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <math.h>
#include "tensor.h"

int softmax_forward(TENSOR *tensor, TENSOR *a);
int softmax_backward(TENSOR *tensor, TENSOR *a);
int gpu_softmax_forward(TENSOR *tensor, TENSOR *a);
int cpu_softmax_forward(TENSOR *tensor, TENSOR *a);

int softmax_forward(TENSOR *target, TENSOR *a)
{
    if (target->run_on_gpu)
    {
        return gpu_softmax_forward(target, a);
    }

    return cpu_softmax_forward(target, a);
}

int softmax_backward(TENSOR *tensor, TENSOR *a)
{
    // backward softmax is disabled
    return 0;
}

int cpu_softmax_forward(TENSOR *target, TENSOR *a)
{
    for (int i = 0; i < a->mat_shape->x; i++)
    {
        double sum = 0;
        for (int j = 0; j < a->mat_shape->y; j++)
        {
            sum += exp(a->data[i * a->mat_shape->y + j]);
        }
        for (int j = 0; j < a->mat_shape->y; j++)
        {
            int index = i * a->mat_shape->y + j;
            target->data[index] = exp(a->data[index]) / sum;
        }
    }

    return 0;
}

#endif
