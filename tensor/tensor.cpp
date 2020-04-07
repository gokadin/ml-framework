#include "tensor.h"

TENSOR *alloc_tensor() {
    return (TENSOR*)malloc(sizeof(TENSOR));
}

void free_tensor(TENSOR *p) {
    free(p);
}

