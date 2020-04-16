#include "tensor.h"

int matmul(TENSOR *a, TENSOR *b, TENSOR *target);
int matmul_backward(TENSOR *tensor, TENSOR *a, TENSOR *b);
