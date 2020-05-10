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
    free(tensor);
}
