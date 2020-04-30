#ifndef LINEAR_H
#define LINEAR_H

#include "tensor.h"

int linear_forward(TENSOR *target);
int linear_backward(TENSOR *target);
int gpu_linear(TENSOR *a, TENSOR *x, TENSOR *b, TENSOR *target);
int cpu_linear_forward(TENSOR *a, TENSOR *x, TENSOR *b, TENSOR *target);
int cpu_linear_backward(TENSOR *target, TENSOR *a, TENSOR *x);
SHAPE linear_target_shape(TENSOR *tensor);

OP *alloc_linear(TENSOR *a, TENSOR *x, TENSOR *b)
{
    OP *op = (OP*)malloc(sizeof(OP));

    op->forward = linear_forward;
    op->backward = linear_backward;
    op->target_shape = linear_target_shape;

    op->operands = malloc(sizeof(TENSOR*) * 3);
    op->operands[0] = a;
    op->operands[1] = x;
    op->operands[2] = b;

    return op;
}

SHAPE linear_target_shape(TENSOR *tensor)
{
    SHAPE shape;
    shape.x = tensor->op->operands[0]->mat_shape->y;
    shape.y = tensor->op->operands[1]->mat_shape->x;
    shape.size = shape.x * shape.y;
    return shape;
}

int linear_forward(TENSOR *target)
{
    if (target->op->operands[0]->mat_shape->y != target->op->operands[1]->mat_shape->x)
    {
        return 1;
    }

    if (target->run_on_gpu)
    {
        return gpu_linear(target->op->operands[0], target->op->operands[1], target->op->operands[2], target);
    }

    return cpu_linear_forward(target->op->operands[0], target->op->operands[1], target->op->operands[2], target);
}

// TODO
int linear_backward(TENSOR *target)
{
	//oa.b.SetGradient(mat.Sum(mat.NewMat32f(mat.WithShape(tensor.Shape().X, tensor.Shape().Y), tensor.GradientToFloat32()), 0).Data())
	//C.matmul_backward(tensor._tensor, oa.a._tensor, oa.x._tensor)
    return gpu_matmul_backward(target, target->op->operands[0], target->op->operands[1]);
}

// TODO
int cpu_linear_forward(TENSOR *a, TENSOR *x, TENSOR *b, TENSOR *target)
{
//    for (int i = 0; i < target->mat_shape->size; i++)
//    {
//        target->data[i] = a->data[i] + b->data[i];
//    }

    return 0;
}

// TODO
int cpu_linear_backward(TENSOR *target, TENSOR *a, TENSOR *x)
{
//    for (int i = 0; i < target->grad_shape->size; i++)
//    {
//        a->grad[i] = target->grad[i];
//        b->grad[i] = target->grad[i];
//    }

    return 0;
}

#endif
