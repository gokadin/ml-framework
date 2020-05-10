#ifndef OP_H
#define OP_H

typedef struct {
    struct TENSOR **operands;
    SHAPE (*target_shape)(struct TENSOR *target);
    int (*forward)(struct TENSOR *target);
    int (*backward)(struct TENSOR *target);
} OP;

#endif
