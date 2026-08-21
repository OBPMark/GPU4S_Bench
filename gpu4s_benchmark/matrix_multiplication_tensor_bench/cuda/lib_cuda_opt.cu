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
    __shared__ bench_t A_tile[BLOCK_SIZE*BLOCK_SIZE];
    __shared__ bench_t B_tile[BLOCK_SIZE*BLOCK_SIZE];

    
    unsigned int i = blockIdx.x *  BLOCK_SIZE + threadIdx.x;
    unsigned int j = blockIdx.y *  BLOCK_SIZE + threadIdx.y;
    
    
    bench_t acumulated = 0;
    unsigned int idx = 0;
    // load memory
    for (unsigned int sub = 0; sub < gridDim.x; ++sub)
    {
        
        idx = i * n + sub * BLOCK_SIZE + threadIdx.y;

        if(idx >= m*n)
        {
            A_tile[threadIdx.x * BLOCK_SIZE+ threadIdx.y] = 0;
        }
        else
        {   
            A_tile[threadIdx.x * BLOCK_SIZE + threadIdx.y] = A[idx];
        }
        idx = (sub * BLOCK_SIZE + threadIdx.x) * w + j;

        if (idx >= m*w)
        {
            B_tile[threadIdx.x * BLOCK_SIZE +  threadIdx.y] = 0;
        }
        else
        {   
            B_tile[threadIdx.x* BLOCK_SIZE + threadIdx.y] = B[idx];
        }
        __syncthreads();
        for (unsigned int k = 0; k < BLOCK_SIZE; ++k)
        {
            acumulated +=  A_tile[threadIdx.x*BLOCK_SIZE + k] * B_tile[k*BLOCK_SIZE + threadIdx.y];
        }
        __syncthreads();

    }
    if (i < n && j < w)
    {
        
        C[i *n + j] = acumulated;
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

    matrix_multiplication_kernel<<<dimGrid, dimBlock>>>(deviceObj->d_A, deviceObj->d_B, deviceObj->d_C, n, m, w);
    
    // profilling end 
    cudaEventRecord(*deviceObj->stop);
    cudaDeviceSynchronize(); 
    kernelCLK.end();

    // store the kernel time
    deviceObj->elapsed_time = kernelCLK.getElapsedMS();
}

