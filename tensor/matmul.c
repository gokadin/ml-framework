#include "matmul.h"

OP *alloc_matmul(TENSOR *a, TENSOR *b)
{
    OP *op = (OP*)malloc(sizeof(OP));

    op->forward = matmul_forward;
    op->backward = matmul_backward;
    op->target_shape = matmul_target_shape;

    op->operands = malloc(sizeof(TENSOR*) * 2);
    op->operands[0] = a;
    op->operands[1] = b;

    return op;
}

SHAPE matmul_target_shape(TENSOR *tensor)
{
    SHAPE shape;
    shape.x = tensor->op->operands[0]->mat_shape->x;
    shape.y = tensor->op->operands[1]->mat_shape->y;
    shape.size = shape.x * shape.y;
    return shape;
}

int matmul_forward(TENSOR *target)
{
    if (target->op->operands[0]->mat_shape->y != target->op->operands[1]->mat_shape->x)
    {
        return 1;
    }

    if (target->run_on_gpu)
    {
        return gpu_matmul_forward(target->op->operands[0], target->op->operands[1], target);
    }

    return cpu_matmul_forward(target->op->operands[0], target->op->operands[1], target);
}

int matmul_backward(TENSOR *target)
{
    if (target->run_on_gpu)
    {
        return gpu_matmul_backward(target, target->op->operands[0], target->op->operands[1]);
    }

    return cpu_matmul_backward(target, target->op->operands[0], target->op->operands[1]);
}

int cpu_matmul_forward(TENSOR *a, TENSOR *b, TENSOR *target)
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
int cpu_matmul_backward(TENSOR *target, TENSOR *a, TENSOR *x)
{
    return gpu_matmul_backward(target, target->op->operands[0], target->op->operands[1]);
//    return 0;
}
