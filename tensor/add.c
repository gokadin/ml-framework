#include "add.h"

int add_forward(TENSOR *tensor, TENSOR *a, TENSOR *b)
{
    if (tensor->run_on_gpu)
    {
        return gpu_add_forward(tensor, a, b);
    }

    return cpu_add_forward(tensor, a, b);
}

int add_backward(TENSOR *tensor, TENSOR *a, TENSOR *b)
{
    return cpu_add_backward(tensor, a, b);
}

int cpu_add_forward(TENSOR *tensor, TENSOR *a, TENSOR *b)
{
    for (int i = 0; i < tensor->mat_size; i++)
    {
        tensor->data[i] = a->data[i] + b->data[i];
    }

    return 0;
}

int cpu_add_backward(TENSOR *tensor, TENSOR *a, TENSOR *b)
{
    for (int i = 0; i < tensor->grad_size; i++)
    {
        a->grad[i] = tensor->grad[i];
        b->grad[i] = tensor->grad[i];
    }

    return 0;
}

