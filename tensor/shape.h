#ifndef SHAPE_H
#define SHAPE_H

#include <stdbool.h>

typedef struct SHAPE {
    int x;
    int y;
    int size;
} SHAPE;

bool shapes_equal(SHAPE *a, SHAPE *b);

#endif
