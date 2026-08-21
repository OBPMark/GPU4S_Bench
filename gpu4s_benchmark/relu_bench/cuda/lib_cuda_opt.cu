#include "../benchmark_library.h"
#include "math.h"

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void
relu_kernel(const bench_t *A, bench_t *B, const int size)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    bench_t threshold = 0;
    if (i  < (size * size)){
        
        #ifdef INT
        B[i] = max(threshold, A[i]);
        #elif FLOAT
        B[i] = max(threshold, A[i]);
        #else
        B[i] = fmax(threshold, A[i]);
        //}
        
        #endif
    }
}

void execute_kernel(GraficCommon* device_object, unsigned int n, unsigned int m,unsigned int w){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(ceil(float((n*n))/(dimBlock.x)));
    // kernel time execution
    Clock kernelCLK;

    // profilling start 
    kernelCLK.start();
    cudaEventRecord(*deviceObj->start);

    relu_kernel<<<dimGrid, dimBlock>>>(deviceObj->d_A, deviceObj->d_B, n);

    // profilling end 
    cudaEventRecord(*deviceObj->stop);
    cudaDeviceSynchronize(); 
    kernelCLK.end();

    // store the kernel time
    deviceObj->elapsed_time = kernelCLK.getElapsedMS();
}
