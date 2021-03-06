#include "matmul.h"

int matmul_forward(TENSOR *target, TENSOR *a, TENSOR *b)
{
    if (target->run_on_gpu)
    {
        return gpu_matmul_forward(target, a, b);
    }

    return cpu_matmul_forward(target, a, b);
}

int matmul_backward(TENSOR *target, TENSOR *a, TENSOR *b)
{
    if (target->run_on_gpu)
    {
        return gpu_matmul_backward(target, a, b);
    }

    return cpu_matmul_backward(target, a, b);
}

int cpu_matmul_forward(TENSOR *target, TENSOR *a, TENSOR *b)
{
    for (int i = 0; i < a->mat_shape->x; i++)
    {
        for (int j = 0; j < b->mat_shape->y; j++)
        {
            for (int k = 0; k < b->mat_shape->x; k++)
            {
                target->data[i * b->mat_shape->y + j] += a->data[i * b->mat_shape->x + k] * b->data[k * b->mat_shape->y + j];
            }
        }
    }

    return 0;
}

// TODO
int cpu_matmul_backward(TENSOR *tensor, TENSOR *a, TENSOR *b)
{
    return gpu_matmul_backward(tensor, a, b);
//    return 0;
}
