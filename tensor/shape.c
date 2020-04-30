#include "shape.h"

bool shapes_equal(SHAPE *a, SHAPE *b)
{
    return a->x == b->x && a->y == b->y;
}
