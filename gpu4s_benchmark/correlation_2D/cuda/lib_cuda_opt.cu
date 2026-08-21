#include "../benchmark_library.h"

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */

__global__ void
mean_matrices(const bench_t *A,const bench_t *B,result_bench_t *mean_A ,result_bench_t *mean_B ,const int n)
{   
    unsigned int size = n;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int tid_x = threadIdx.x;
    unsigned int tid_y = threadIdx.y;
  
    __shared__ bench_t shared_data_A[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ bench_t shared_data_B[BLOCK_SIZE * BLOCK_SIZE];

    if (i < size && j < size){

        shared_data_A[tid_x*blockDim.y + tid_y] = A[i*size + j];
        shared_data_B[tid_x*blockDim.y + tid_y] = B[i*size + j];
    } else {
        shared_data_A[tid_x*blockDim.y + tid_y] = 0;
        shared_data_B[tid_x*blockDim.y + tid_y] = 0;
    }

    // sinc theads
    __syncthreads();

    // --- Reduce Y-axis ---
    for(unsigned int s_y = blockDim.y/2; s_y > 0; s_y >>= 1){
        if (tid_y < s_y)
        {
            shared_data_A[tid_x * blockDim.y + tid_y] += shared_data_A[tid_x * blockDim.y + tid_y + s_y];
            shared_data_B[tid_x * blockDim.y + tid_y] += shared_data_B[tid_x * blockDim.y + tid_y + s_y];
        }
        __syncthreads();
    }

    // --- Reduce X-axis ---
    for(unsigned int s_x = blockDim.x/2; s_x > 0; s_x >>= 1 ){
        if(tid_x < s_x && tid_y == 0)
        {
            shared_data_A[tid_x * blockDim.y] += shared_data_A[(tid_x + s_x) * blockDim.y];
            shared_data_B[tid_x * blockDim.y] += shared_data_B[(tid_x + s_x) * blockDim.y];
        }
        __syncthreads();
        
    }

    if( tid_x == 0 && tid_y == 0)
    { 
        atomicAdd(mean_A, shared_data_A[0]);
        atomicAdd(mean_B, shared_data_B[0]);
    }
}


__global__ void
correlation_2D(const bench_t *A,const bench_t *B, result_bench_t *R, result_bench_t *mean_A ,result_bench_t *mean_B, result_bench_t *acumulate_value_a_b, result_bench_t *acumulate_value_a_a, result_bench_t *acumulate_value_b_b,const int n){
    unsigned int size = n;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int tid_x = threadIdx.x;
    unsigned int tid_y = threadIdx.y;

    result_bench_t mean_a_matrix =  *mean_A / (n * n);
    result_bench_t mean_b_matrix =  *mean_B / (n * n);
    
    __shared__ bench_t shared_data_A_B[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ bench_t shared_data_A_A[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ bench_t shared_data_B_B[BLOCK_SIZE * BLOCK_SIZE];

    if (i < size && j < size){
        result_bench_t result_mean_a = 0;
        result_bench_t result_mean_b = 0;
        result_mean_a = A[i*size+j] - mean_a_matrix;
        result_mean_b = B[i*size+j] - mean_b_matrix;
        shared_data_A_B[tid_x*blockDim.y + tid_y] = result_mean_a * result_mean_b;
        shared_data_A_A[tid_x*blockDim.y + tid_y] = result_mean_a * result_mean_a;
        shared_data_B_B[tid_x*blockDim.y + tid_y] = result_mean_b * result_mean_b;

        // first get the final value  in A (A - mean(a)) and in B (B - mean(b))
        __syncthreads();
        
        for(unsigned int s_y = blockDim.y/2; s_y > 0; s_y >>= 1)
        {
            if (tid_y < s_y)
            {
                shared_data_A_B[tid_x * blockDim.y + tid_y] += shared_data_A_B[tid_x * blockDim.y + tid_y + s_y];
                shared_data_A_A[tid_x * blockDim.y + tid_y] += shared_data_A_A[tid_x * blockDim.y + tid_y + s_y];
                shared_data_B_B[tid_x * blockDim.y + tid_y] += shared_data_B_B[tid_x * blockDim.y + tid_y + s_y];
            }
            __syncthreads();
        }
        for(unsigned int s_x = blockDim.x/2; s_x > 0; s_x >>= 1 )
        {
            if(tid_x < s_x)
            {
                shared_data_A_B[tid_x * blockDim.y] += shared_data_A_B[(tid_x + s_x) * blockDim.y];
                shared_data_A_A[tid_x * blockDim.y] += shared_data_A_A[(tid_x + s_x) * blockDim.y];
                shared_data_B_B[tid_x * blockDim.y] += shared_data_B_B[(tid_x + s_x) * blockDim.y];
            }
            __syncthreads();
            
        }

        if( tid_x == 0 && tid_y == 0)
        { 
            atomicAdd(acumulate_value_a_b, shared_data_A_B[0]);
            atomicAdd(acumulate_value_a_a, shared_data_A_A[0]);
            atomicAdd(acumulate_value_b_b, shared_data_B_B[0]);
        }
 
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
    cudaEventRecord(*deviceObj->start);

    mean_matrices<<<dimGrid,dimBlock>>>(deviceObj->d_A, deviceObj->d_B, deviceObj->mean_A, deviceObj->mean_B , n);
    correlation_2D<<<dimGrid,dimBlock>>>(deviceObj->d_A, deviceObj->d_B, deviceObj->d_R, deviceObj->mean_A, deviceObj->mean_B,deviceObj->acumulate_value_a_b, deviceObj->acumulate_value_a_a, deviceObj->acumulate_value_b_b, n);

    // profilling end 
    cudaEventRecord(*deviceObj->stop);
    cudaDeviceSynchronize(); 
    kernelCLK.end();

    // store the kernel time
    deviceObj->elapsed_time = kernelCLK.getElapsedMS();
}

