#ifndef GRAPH_H
#define GRAPH_H

#include <stdlib.h>
#include "tensor.h"

typedef struct {
    float x;
} GRAPH;

void testgraph(TENSOR *a, TENSOR *b, TENSOR *tensor);

#endif
