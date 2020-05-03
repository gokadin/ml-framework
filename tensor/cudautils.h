#define checkCudaErr(err)  __checkCudaErr (err, __FILE__, __LINE__)
inline void __checkCudaErr(cudaError err, const char *file, const int line )
{
    if(cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : CUDA runtime error %d: %s.\n", file, line, (int)err, cudaGetErrorString(err));
        exit(-2);
    }
}