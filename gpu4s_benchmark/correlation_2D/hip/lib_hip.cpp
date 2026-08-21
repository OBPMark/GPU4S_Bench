#include "../benchmark_library.h"


/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */

 __global__ void
mean_matrices (const bench_t *A,const bench_t *B,result_bench_t *mean_A ,result_bench_t *mean_B ,const int n){
    unsigned int size = n;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < size && j < size){
        atomicAdd(mean_A, A[i*size+j]);
        atomicAdd(mean_B, B[i*size+j]);
    }
}

__global__ void
correlation_2D(const bench_t *A,const bench_t *B, result_bench_t *R, result_bench_t *mean_A ,result_bench_t *mean_B, result_bench_t *acumulate_value_a_b, result_bench_t *acumulate_value_a_a, result_bench_t *acumulate_value_b_b,const int n){
    unsigned int size = n;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    result_bench_t mean_a_matrix =  *mean_A / (n * n);
	result_bench_t mean_b_matrix =  *mean_B / (n * n);
    if (i < size && j < size){
        // first get the final value  in A (A - mean(a)) and in B (B - mean(b))
        result_bench_t result_mean_a = 0;
        result_bench_t result_mean_b = 0;
        result_mean_a = A[i*size+j] - mean_a_matrix;
        result_mean_b = B[i*size+j] - mean_b_matrix;
        atomicAdd(acumulate_value_a_b, result_mean_a * result_mean_b);
        atomicAdd(acumulate_value_a_a, result_mean_a * result_mean_a);
        atomicAdd(acumulate_value_b_b, result_mean_b * result_mean_b);
        // final calculation
        //__syncthreads(); //TODO CHECK
        //if (i == 0 && j == 0){
            //*R = (result_bench_t)(*acumulate_value_a_b / (result_bench_t)(sqrt(*acumulate_value_a_a * *acumulate_value_b_b)));
            //printf("Rvalue %f\n", *R);
        //}
    }

}

void execute_kernel(GraficCommon* device_object, unsigned int n){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    dim3 dimGrid(ceil(float(n)/dimBlock.x),ceil(float(n)/dimBlock.y));
    // kernel time execution
    Clock kernelCLK;

    // profilling start 
    kernelCLK.start();
    (void)hipEventRecord(*deviceObj->start);
    
    hipLaunchKernelGGL(mean_matrices, dim3(dimGrid), dim3(dimBlock), 0, 0, deviceObj->d_A, deviceObj->d_B, deviceObj->mean_A, deviceObj->mean_B , n);
    hipLaunchKernelGGL(correlation_2D, dim3(dimGrid), dim3(dimBlock), 0, 0, deviceObj->d_A, deviceObj->d_B, deviceObj->d_R, deviceObj->mean_A, deviceObj->mean_B,deviceObj->acumulate_value_a_b, deviceObj->acumulate_value_a_a, deviceObj->acumulate_value_b_b, n);

    // profilling end 
    (void)hipEventRecord(*deviceObj->stop);
    hipDeviceSynchronize(); 
    kernelCLK.end();

    // store the kernel time
    deviceObj->elapsed_time = kernelCLK.getElapsedMS();
}

