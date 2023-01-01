#include <cstdio>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32
#define DIVUP(m, n) ((m + n - 1) / n)


__global__ void two_sum_kernel(const float* a, const float* b, float * c, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        c[idx] = a[idx] + b[idx];
    }
}


void two_sum_launcher(const float* a, const float* b, float* c, int n){
    dim3 blockSize(DIVUP(n, THREADS_PER_BLOCK));
    dim3 threadSize(THREADS_PER_BLOCK);
    two_sum_kernel<<<blockSize, threadSize>>>(a, b, c, n);
}