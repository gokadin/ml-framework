#ifndef EXPAND_H
#define EXPAND_H

#include "tensor.h"

int expand_forward(TENSOR *target, TENSOR *a, int axis, int copies);
int cpu_expand_forward(TENSOR *target, TENSOR *a, int axis, int copies);
int gpu_expand_forward(TENSOR *target, TENSOR *a, int axis, int copies);

int expand_forward(TENSOR *target, TENSOR *a, int axis, int copies)
{
    if (target->run_on_gpu)
    {
        return gpu_expand_forward(target, a, axis, copies);
    }

    return cpu_expand_forward(target, a, axis, copies);
}

int cpu_expand_forward(TENSOR *target, TENSOR *a, int axis, int copies)
{
    if (axis == 0)
    {
        for (int i = 0; i < copies; i++)
        {
            for (int j = 0; j < a->mat_shape[1]; j++)
            {
                target->data[i * a->mat_shape[1] + j] = a->data[j];
            }
        }

	    return 0;
    }

    if (axis == 1)
    {
        for (int i = 0; i < a->mat_shape[0]; i++)
        {
            for (int j = 0; j < copies; j++)
            {
                target->data[i * copies + j] = a->data[i];
            }
        }

	    return 0;
    }

    return 1;
}

#endif
