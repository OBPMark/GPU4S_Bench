#include "../benchmark_library.h"


/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */

 __global__ void
covolution_kernel(const bench_t *A, bench_t *B, const bench_t *kernel,const int n, const int m, const int w, const int kernel_size)
{
    unsigned int size = n;
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    int kernel_rad = kernel_size / 2;

    bench_t sum = 0;

    if (x < size && y < size)
    {
        for(int i = -kernel_rad; i <= kernel_rad; ++i) // loop over kernel_rad  -1 to 1 in kernel_size 3 
            {
                for(int j = -kernel_rad; j <= kernel_rad; ++j){
                    // get value
                    bench_t value = 0;
                    
                    if (i + x < 0 || j + y < 0)
                    {
                        value = 0;
                        //printf("ENTRO %d %d\n", i + x , j + y);
                    }
                    else if ( i + x > size - 1 || j + y > size - 1)
                    {
                        value = 0;
                        //printf("ENTRO UPPER%d %d\n", i + x , j + y);
                    }
                    else
                    {
                        value = A[(x + i)*size+(y + j)];
                    }
                    //printf("ACHIVED position  %d %d value %f\n", (x + i) , (y + j), value);
                    sum += value * kernel[(i+kernel_rad)* kernel_size + (j+kernel_rad)];
                }
            }
            
    B[x*size+y ] = sum;
    }
    
}

void execute_kernel(GraficCommon* device_object, unsigned int n, unsigned int m,unsigned int w, unsigned int kernel_size){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(ceil(float(n)/dimBlock.x), ceil(float(m)/dimBlock.y));
    // kernel time execution
    Clock kernelCLK;

    // profilling start 
    kernelCLK.start();
    (void)hipEventRecord(*deviceObj->start);

    hipLaunchKernelGGL((covolution_kernel), dim3(dimGrid), dim3(dimBlock), 0, 0, deviceObj->d_A, deviceObj->d_B, deviceObj->kernel, n, m, w, kernel_size);
    
    // profilling end 
    (void)hipEventRecord(*deviceObj->stop);
    hipDeviceSynchronize(); 
    kernelCLK.end();

    // store the kernel time
    deviceObj->elapsed_time = kernelCLK.getElapsedMS();
}



