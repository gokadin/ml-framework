#include "tensor.h"

TENSOR *alloc_tensor()
{
    TENSOR *tensor = (TENSOR*)malloc(sizeof(TENSOR));
    tensor->run_on_gpu = true;
    return tensor;
}

void free_tensor(TENSOR *tensor)
{
    if (tensor->op != NULL)
    {
        free(tensor->op);
    }

    free(tensor);
}

void set_mat_shape(TENSOR *tensor, int x, int y)
{
    tensor->mat_shape.x = x;
    tensor->mat_shape.y = y;
    tensor->mat_shape.size = x * y;
}

void set_grad_shape(TENSOR *tensor, int x, int y)
{
    tensor->grad_shape.x = x;
    tensor->grad_shape.y = y;
    tensor->grad_shape.size = x * y;
}

int forward(TENSOR *tensor)
{
    return tensor->op->forward(tensor);
}

int backward(TENSOR *tensor)
{
    return tensor->op->backward(tensor);
}
