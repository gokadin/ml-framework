#ifndef SUM_H
#define SUM_H

#include "tensor.h"

int gpu_sum_forward(TENSOR *a, int axis, TENSOR *target);

#endif
