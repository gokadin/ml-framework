#include "tensor.h"

int gpu_matmul(TENSOR *a, TENSOR *b, TENSOR *target);
int gpu_matmul_backward(TENSOR *tensor, TENSOR *a, TENSOR *b);
