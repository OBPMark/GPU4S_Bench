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
{   unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;
    bench_t value = 0;
    __shared__ bench_t shared_data[BLOCK_SIZE];
    if (i  < (size * size)){
        
        #ifdef INT
        value = exp(A[i]);
        #elif FLOAT
        value = expf(A[i]);
        #else
        value = exp(A[i]);
        #endif

        shared_data[tid] = value;
        B[i] = value;
        // sinc theads
        __syncthreads();
        for (unsigned int s=blockDim.x/2; s>0; s>>=1) 
        {
            if (tid < s)  
            {
                shared_data[tid] += shared_data[tid + s];
            }
        __syncthreads();
        }
        if (tid == 0){
            atomicAdd(sum_d_B, shared_data[0]);
        }    
    }
}
__global__ void
softmax_finish_kernel(bench_t *B, bench_t *sum_d_B,const int size)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i  < (size * size)){
        B[i] = (B[i]/(*sum_d_B));
    }
}

void execute_kernel(GraficCommon* device_object, unsigned int n, unsigned int m,unsigned int w){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    dim3 dimBlock, dimGrid;
    dimBlock = dim3(BLOCK_SIZE);
    dimGrid = dim3(ceil(float((n*n))/(dimBlock.x)));
    // kernel time execution
    Clock kernelCLK;

    // profilling start 
    kernelCLK.start();
    (void)hipEventRecord(*deviceObj->start);

    hipLaunchKernelGGL((softmax_kernel), dim3(dimGrid), dim3(dimBlock), 0, 0, deviceObj->d_A, deviceObj->d_B, deviceObj->sum_d_B, n);
    hipLaunchKernelGGL((softmax_finish_kernel), dim3(dimGrid), dim3(dimBlock), 0, 0, deviceObj->d_B, deviceObj->sum_d_B, n);
    
    // profilling end 
    (void)hipEventRecord(*deviceObj->stop);
    hipDeviceSynchronize(); 
    kernelCLK.end();

    // store the kernel time
    deviceObj->elapsed_time = kernelCLK.getElapsedMS();
}