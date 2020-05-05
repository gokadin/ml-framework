#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#define checkCudaErr(err)  __checkCudaErr (err, __FILE__, __LINE__)
inline void __checkCudaErr(cudaError err, const char *file, const int line )
{
    if(cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : CUDA runtime error %d: %s.\n", file, line, (int)err, cudaGetErrorString(err));
        exit(-2);
    }
}

#define checkCudaKernelErr(kernel_name, block_size, grid_size)  __checkCudaKernelErr (kernel_name, block_size, grid_size, __FILE__, __LINE__)
inline void __checkCudaKernelErr(const char *kernel_name, dim3 block_size, dim3 grid_size, const char *file, const int line )
{
    cudaError err = cudaPeekAtLastError();
    if(cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : CUDA runtime error %d: Kernel %s failed with block size %dx%dx%d and grid size %dx%dx%d: %s.\n", file, line, (int)err, kernel_name, block_size.x, block_size.y, block_size.z, grid_size.x, grid_size.y, grid_size.z, cudaGetErrorString(err));
        exit(-2);
    }
}

#endif
