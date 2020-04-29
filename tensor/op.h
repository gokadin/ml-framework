#ifndef OP_H
#define OP_H

typedef struct {
    struct TENSOR *dependencies;
    int (*forward)(struct TENSOR *target);
    int (*backward)(struct TENSOR *target);
} OP;

#endif
