#include "tensor.h"

TENSOR *alloc_tensor()
{
    return (TENSOR*)malloc(sizeof(TENSOR));
}

void free_tensor(TENSOR *tensor)
{
//    free(tensor->graph.dependencies);
    free(tensor);
}

//void set_op(OP_TYPE op_type, TENSOR *tensor, TENSOR *a, TENSOR *b)
//{
//    tensor->graph.op_type = op_type;
//    tensor->graph.dependencies = (TENSOR*)malloc(2 * sizeof(TENSOR));
//    tensor->graph.dependencies[0] = a[0];
//    tensor->graph.dependencies[1] = b[0];
//}

//void forward(TENSOR *tensor)
//{
//	add(tensor->graph.dependencies[0].data, tensor->graph.dependencies[1].data, tensor->data, tensor->mat_shape.x * tensor->mat_shape.y);
//}
