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


void aux_execute_kernel(GraficCommon* device_object, int64_t size, bench_cuda_complex *d_A, bench_cuda_complex *d_B, cufftHandle *plan){    
    //bench_cuda_complex* d_B = deviceObj->d_B;
    
    #ifdef FLOAT
    cufftPlan1d(plan, size/2, CUFFT_C2C, 1);
    cufftExecC2C(*plan, (bench_cuda_complex *)d_A, (bench_cuda_complex *)d_B, CUFFT_FORWARD);
    #else 
    cufftPlan1d(plan, size/2, CUFFT_Z2Z, 1);
    cufftExecZ2Z(*plan, (bench_cuda_complex *)d_A, (bench_cuda_complex *)d_B, CUFFT_FORWARD);
    #endif
}

void execute_kernel(GraficCommon* device_object, int64_t window, int64_t size){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    bench_cuda_complex* d_A = (bench_cuda_complex*)deviceObj->d_A;
    bench_cuda_complex* d_B = (bench_cuda_complex*)deviceObj->d_B;
    cufftHandle plan;
    // kernel time execution
    Clock kernelCLK;

    // profilling start 
    kernelCLK.start();
    cudaEventRecord(*deviceObj->start);

    for (unsigned int i = 0; i < (size * 2  - window + 1); i+=1){
        aux_execute_kernel(device_object, window, d_A, d_B, &plan);
        d_B += window;
        ++d_A;

    }

    // profilling end 
    cudaEventRecord(*deviceObj->stop);
    cudaDeviceSynchronize(); 
    kernelCLK.end();

    // store the kernel time
    deviceObj->elapsed_time = kernelCLK.getElapsedMS();

    cufftDestroy(plan);
}

