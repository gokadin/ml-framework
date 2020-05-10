#ifndef MATMUL_CUH
#define MATMUL_CUH

__global__ void matmul_a_grad(float *cg, float *b, float *ag, int m, int n, int k);
__global__ void matmul_b_grad(float *cg, float *a, float *bg, int m, int n, int k);

#endif
