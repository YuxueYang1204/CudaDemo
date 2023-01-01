#include <cstdio>
#include <cassert>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32
#define DIVUP(m, n) ((m + n - 1) / n)

template <typename T>
__device__ inline T warpReduceSum(T sum, int blockSize) {
    if (blockSize >= 32)
        sum += __shfl_down_sync(0xffffffff, sum, 16);
    if (blockSize >= 16)
        sum += __shfl_down_sync(0xffffffff, sum, 8);
    if (blockSize >= 8)
        sum += __shfl_down_sync(0xffffffff, sum, 4);
    if (blockSize >= 4)
        sum += __shfl_down_sync(0xffffffff, sum, 2);
    if (blockSize >= 2)
        sum += __shfl_down_sync(0xffffffff, sum, 1);
    return sum;
}


__global__ void sum_reduce_kernel(const float* array, float* ans, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    __shared__ float warpSum[WARP_SIZE];
    float sum = (idx < n) ? array[idx] : 0;
    int valid_warps = DIVUP(blockDim.x, WARP_SIZE);
    assert((valid_warps & valid_warps - 1) == 0); // valid_warps must be the power of 2
    int laneIdx = tid % WARP_SIZE;
    int warpIdx = tid / WARP_SIZE;
    sum = warpReduceSum(sum, blockDim.x);
    if (laneIdx == 0){
        warpSum[warpIdx] = sum;
    }
    __syncthreads();
    sum = (tid < valid_warps) ? warpSum[laneIdx] : 0;
    if (warpIdx == 0){
        sum = warpReduceSum(sum, valid_warps);
    }
    if (tid == 0){
        atomicAdd(ans, sum);
    }
}


void sum_reduce_launcher(const float* array, float* ans, int n){
    dim3 blockSize(DIVUP(n, THREADS_PER_BLOCK));
    dim3 threadSize(THREADS_PER_BLOCK);
    sum_reduce_kernel<<<blockSize, threadSize>>>(array, ans, n);
}