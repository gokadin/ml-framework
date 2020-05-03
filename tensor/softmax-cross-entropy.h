#ifndef SOFTMAX_CROSS_ENTROPY_H
#define SOFTMAX_CROSS_ENTROPY_H

#include <math.h>
#include "tensor.h"

OP *alloc_sce(TENSOR *a, TENSOR *b);
SHAPE sce_target_shape(TENSOR *tensor);
int sce_forward(TENSOR *target);
int sce_backward(TENSOR *tensor);
int gpu_sce_forward(TENSOR *a, TENSOR *b, TENSOR *target);
int cpu_sce_forward(TENSOR *a, TENSOR *b, TENSOR *target);
int gpu_sce_backward(TENSOR *tensor, TENSOR *a, TENSOR *b);
int cpu_sce_backward(TENSOR *tensor, TENSOR *a, TENSOR *b);

OP *alloc_sce(TENSOR *a, TENSOR *b)
{
    OP *op = (OP*)malloc(sizeof(OP));

    op->forward = sce_forward;
    op->backward = sce_backward;
    op->target_shape = sce_target_shape;

    op->operands = malloc(sizeof(TENSOR*) * 2);
    op->operands[0] = a;
    op->operands[1] = b;

    return op;
}

SHAPE sce_target_shape(TENSOR *tensor)
{
    SHAPE shape;
    shape.x = 1;
    shape.y = 1;
    shape.size = 1;
    return shape;
}

int sce_forward(TENSOR *target)
{
    if (!shapes_equal(target->op->operands[0]->mat_shape, target->op->operands[1]->mat_shape))
    {
        return 1;
    }

    if (target->run_on_gpu)
    {
        // TODO need to do softmax first
        return gpu_sce_forward(target->op->operands[0], target->op->operands[1], target);
    }

    return cpu_sce_forward(target->op->operands[0], target->op->operands[1], target);
}

int sce_backward(TENSOR *tensor)
{
    return cpu_sce_backward(tensor, tensor->op->operands[0], tensor->op->operands[1]);
}

int cpu_sce_forward(TENSOR *a, TENSOR *b, TENSOR *target)
{
    float partial[a->mat_shape->size];
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
            partial[index] = log(expf(a->data[index]) / sum) * b->data[index];
        }
    }

    float sum1[a->mat_shape->x];
    for (int i = 0; i < a->mat_shape->x; i++)
    {
        sum1[i] = 0;
        for (int j = 0; j < a->mat_shape->y; j++)
        {
            sum1[i] += a->data[i * a->mat_shape->y + j];
        }
    }

    float sum0 = 0;
    for (int i = 0; i < a->mat_shape->x; i++)
    {
        sum0 += sum1[i];
    }

    target->data[0] = -sum0;

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
