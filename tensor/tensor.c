#include "tensor.h"

TENSOR *alloc_tensor(int id)
{
    TENSOR *tensor = (TENSOR*)malloc(sizeof(TENSOR));
    tensor->id = id;
    tensor->run_on_gpu = true;
    return tensor;
}

void free_tensor(TENSOR *tensor)
{
    free(tensor->op);
    free(tensor);
}

SHAPE calculate_op_shape(struct TENSOR *tensor)
{
    return tensor->op->target_shape(tensor);
}

int forward(TENSOR *tensor)
{
    return tensor->op->forward(tensor);
}

int backward(TENSOR *tensor)
{
    return tensor->op->backward(tensor);
}
