#include "../benchmark_library.h"


/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void
matrix_multiplication_kernel(const bench_t *A,const bench_t *B,  bench_t *C, const int n, const int m, const int w)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < w){
        bench_t acumulated = 0;
        for (unsigned int k_d = 0; k_d < m; ++k_d )
        {
            acumulated += A[i*n+k_d] * B[k_d*w +j];
        }
        C[i*n+j] =  acumulated;
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
    (void)hipEventRecord(*deviceObj->start);

    hipLaunchKernelGGL((matrix_multiplication_kernel), dim3(dimGrid), dim3(dimBlock), 0, 0, deviceObj->d_A, deviceObj->d_B, deviceObj->d_C, n, m, w);
    
    // profilling end 
    (void)hipEventRecord(*deviceObj->stop);
    hipDeviceSynchronize(); 
    kernelCLK.end();

    // store the kernel time
    deviceObj->elapsed_time = kernelCLK.getElapsedMS();
}
