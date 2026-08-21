#include "../benchmark_library.h"

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */

__global__ void
binary_reverse_kernel(const bench_t *A, bench_t *B, const int64_t size, const int group, const int position_off)
{
    
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int position = 0;
    if (id < size)
    {   
        position = (__brev(id) >> (32 - group)) * 2;
        B[(position) + (size * 2 * position_off)] = A[(id *2) + position_off];
        B[position + 1 +  (size * 2 * position_off)] = A[(id *2 + 1) + position_off];
    }
}

__global__ void
fft_kernel( bench_t *B, const int loop, const int inner_loop,const bench_t wr, const bench_t wi, const int64_t size, const int64_t position_off)
{   
    bench_t tempr, tempi;
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i;
    unsigned int j;
    // get I
    i = id *(loop * 2 * 2) + 1 + (inner_loop * 2); 
    j=i+(loop * 2 );

    tempr = wr*B[j-1 + (size * 2 * position_off)] - wi*B[j+ (size * 2 * position_off)];
    tempi = wr * B[j+ (size * 2 * position_off)] + wi*B[j-1+ (size * 2 * position_off)];
    
    B[j-1+ (size * 2 * position_off)] = B[i-1+ (size * 2 * position_off)] - tempr;
    B[j+ (size * 2 * position_off)] = B[i+ (size * 2 * position_off)] - tempi;
    B[i-1+ (size * 2 * position_off)] += tempr;
    B[i+ (size * 2 * position_off)] += tempi;
    
}


void aux_execute_kernel(GraficCommon* device_object, int64_t size, int64_t position){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    size = size / 2;
    dim3 dimBlock_reverse(BLOCK_SIZE);
    dim3 dimGrid_reverse(ceil(float(size)/dimBlock_reverse.x));
    dim3 dimBlock(0);
    dim3 dimGrid(0);
    bench_t wtemp, wpr, wpi, theta, wr, wi;
    // reorder kernel
    binary_reverse_kernel<<<dimGrid_reverse, dimBlock_reverse>>>(deviceObj->d_A, deviceObj->d_B, size, (int64_t)log2(size), position);
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
            fft_kernel<<<dimGrid, dimBlock>>>(deviceObj->d_B, loop, i, wr, wi, size, position);
            // update WR, WI
            wtemp=wr;
            wr += wr*wpr - wi*wpi;
            wi += wi*wpr + wtemp*wpi;
            
        }
        // update loop values
        loop = loop * 2;
        theads = theads / 2;
       
    }
}

void execute_kernel(GraficCommon* device_object, int64_t window, int64_t size){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    // kernel time execution
    Clock kernelCLK;

    // profilling start 
    kernelCLK.start();
    cudaEventRecord(*deviceObj->start);

    for (unsigned int i = 0; i < (size * 2 - window + 1); i+=2){
        aux_execute_kernel(device_object, window, i);
    }
   
    // profilling end 
    cudaEventRecord(*deviceObj->stop);
    cudaDeviceSynchronize(); 
    kernelCLK.end();

    // store the kernel time
    deviceObj->elapsed_time = kernelCLK.getElapsedMS();
}

