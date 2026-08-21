#include "../benchmark_library.h"

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */

__global__ void
binary_reverse_kernel(const bench_t *B, bench_t *Br, const int64_t size, const int group)
{
    
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int position = 0;
    if (id < size)
    {   
        position = (__brev(id) >> (32 - group)) * 2;
        Br[position] = B[id *2];
        Br[position + 1] = B[id *2 + 1];
    }
}

__global__ void
fft_kernel( bench_t *B, const int loop, const int inner_loop,const bench_t wr, const bench_t wi)
{   
    bench_t tempr, tempi;
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i;
    unsigned int j;
    // get I
    i = id *(loop * 2 * 2) + 1 + (inner_loop * 2); 
    j=i+(loop * 2 );

    tempr = wr*B[j-1] - wi*B[j];
    tempi = wr * B[j] + wi*B[j-1];
    
    B[j-1] = B[i-1] - tempr;
    B[j] = B[i] - tempi;
    B[i-1] += tempr;
    B[i] += tempi;
    
}

void execute_kernel(GraficCommon* device_object, int64_t size){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    dim3 dimBlock_reverse(BLOCK_SIZE);
    dim3 dimGrid_reverse(ceil(float(size)/dimBlock_reverse.x));
    dim3 dimBlock(0);
    dim3 dimGrid(0);

    bench_t wtemp, wpr, wpi, theta, wr, wi;

    // kernel time execution
    Clock kernelCLK;

    // profilling start 
    kernelCLK.start();
    cudaEventRecord(*deviceObj->start);

    // reorder kernel
    binary_reverse_kernel<<<dimGrid_reverse, dimBlock_reverse>>>(deviceObj->d_B, deviceObj->d_Br, size, (int64_t)log2(size));
    // Synchronize
    cudaDeviceSynchronize();
    // kernel call
    unsigned int theads = size /2 ;
    unsigned int loop = 1;
    //printf("size %d, dimBlock %d, dimGrid %d\n", size, dimBlock.x, dimGrid.x);
    //size = size << 1;
    while(loop < size ){
        // caluclate values 
        theta = -(M_PI/loop); // check
        wtemp = sin(0.5*theta);
        wpr = -2.0*wtemp*wtemp;
        wpi = sin(theta);
        wr = 1.0;
        wi = 0.0;
        // calculate block size and thead size
        if (theads % BLOCK_SIZE != 0){
            // inferior part
            dimBlock.x = theads;
            dimGrid.x  = 1;
        }
        else{
            // top part
            dimBlock.x = BLOCK_SIZE;
            dimGrid.x  = (unsigned int)(theads/BLOCK_SIZE);
        }
        // launch kernel loop times
        for(unsigned int i = 0; i < loop; ++i){
            //kernel launch 
            fft_kernel<<<dimGrid, dimBlock>>>(deviceObj->d_Br, loop, i, wr, wi);
            // update WR, WI
            wtemp=wr;
            wr += wr*wpr - wi*wpi;
            wi += wi*wpr + wtemp*wpi;
            
        }
        // update loop values
        loop = loop * 2;
        theads = theads / 2;
       
    }
   
    // profilling end 
    cudaEventRecord(*deviceObj->stop);
    cudaDeviceSynchronize(); 
    kernelCLK.end();

    // store the kernel time
    deviceObj->elapsed_time = kernelCLK.getElapsedMS();
}

