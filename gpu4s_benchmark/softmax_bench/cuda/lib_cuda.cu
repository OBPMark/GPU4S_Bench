#include "../benchmark_library.h"
#include "math.h"

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */

__global__ void
softmax_kernel(const bench_t *A, bench_t *B, bench_t *sum_d_B,const int size)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < size && j < size){
        #ifdef INT
        B[i*size+j] = exp(A[i*size+j]);
        #elif FLOAT
        B[i*size+j] = expf(A[i*size+j]);
        #else
        B[i*size+j] = exp(A[i*size+j]);
        #endif

        atomicAdd(sum_d_B, B[i*size+j]);
    }
}

__global__ void
softmax_finish_kernel(bench_t *B, bench_t *sum_d_B,const int size)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < size && j < size){
        B[i*size+j] = (B[i*size+j]/(*sum_d_B));
    }
}

void execute_kernel(GraficCommon* device_object, unsigned int n, unsigned int m,unsigned int w){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(ceil(float(n)/dimBlock.x), ceil(float(m)/dimBlock.y));
    // kernel time execution
    Clock kernelCLK;

    // profilling start 
    kernelCLK.start();
    cudaEventRecord(*deviceObj->start);

    softmax_kernel<<<dimGrid, dimBlock>>>(deviceObj->d_A, deviceObj->d_B, deviceObj->sum_d_B, n);
    softmax_finish_kernel<<<dimGrid, dimBlock>>>(deviceObj->d_B, deviceObj->sum_d_B, n);
    
    // profilling end 
    cudaEventRecord(*deviceObj->stop);
    cudaDeviceSynchronize(); 
    kernelCLK.end();

    // store the kernel time
    deviceObj->elapsed_time = kernelCLK.getElapsedMS();
}