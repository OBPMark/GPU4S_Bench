#include "../benchmark_library.h"
#include "math.h"

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
#define BLOCK_SIZE_PLANE (BLOCK_SIZE * BLOCK_SIZE)
__global__ void
max_pooling_kernel(const bench_t *A, bench_t *B, const int size, const unsigned int stride,  const unsigned int lateral_stride)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    //unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
   
    if (i  < lateral_stride*lateral_stride){
        
        bench_t max_value = A[(((i%lateral_stride) * stride )+ ((i/lateral_stride)*size * stride)) ];
        for(unsigned int x = 0; x < stride; ++x)
        {
            for(unsigned int y = 0; y < stride; ++y)
            {
                //unsigned int position_array = ((((i%lateral_stride) * stride )+ ((i/lateral_stride)*size * stride)) + x)  + ( y * size);
                //printf("max %f,value %f, pos x %d, pos y %d i position %d, final position %d\n", max_value,  A[position_array], x ,y, i, position_array);
                max_value = 
                // --- FIX: use the correct max function depending one the type ---
                #ifdef INT
                    max_value = max(max_value, A[((((i%lateral_stride) * stride )+ ((i/lateral_stride)*size * stride)) + x)  + ( y * size)]);
                #elif FLOAT
                    max_value = fmaxf(max_value, A[((((i%lateral_stride) * stride )+ ((i/lateral_stride)*size * stride)) + x)  + ( y * size)]);
                #elif DOUBLE
                     max_value = fmax(max_value, A[((((i%lateral_stride) * stride )+ ((i/lateral_stride)*size * stride)) + x)  + ( y * size)]);
                #endif     
                
            }
        }
        //printf("i position %d value %f \n", i, max_value);
        B[i] = max_value;
    }
}

void execute_kernel(GraficCommon* device_object, unsigned int n, unsigned int m,unsigned int w, unsigned int stride, unsigned int lateral_stride){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    dim3 dimBlock, dimGrid;
    if(lateral_stride < BLOCK_SIZE_PLANE)
    {
        dimBlock = dim3(lateral_stride*lateral_stride);
        dimGrid = dim3(1);
    }
    else
    {
        dimBlock = dim3(BLOCK_SIZE_PLANE);
        dimGrid = dim3(ceil((lateral_stride*lateral_stride)/dimBlock.x));
    }

    // kernel time execution
    Clock kernelCLK;

    // profilling start 
    kernelCLK.start();
    cudaEventRecord(*deviceObj->start);

    max_pooling_kernel<<<dimGrid, dimBlock>>>(deviceObj->d_A, deviceObj->d_B, n, stride, lateral_stride);
    
    // profilling end 
    cudaEventRecord(*deviceObj->stop);
    cudaDeviceSynchronize(); 
    kernelCLK.end();

    // store the kernel time
    deviceObj->elapsed_time = kernelCLK.getElapsedMS();
}
