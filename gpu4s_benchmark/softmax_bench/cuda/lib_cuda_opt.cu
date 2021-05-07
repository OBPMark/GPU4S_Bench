#include "../benchmark_library.h"
#include "math.h"

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
//#define BLOCK_SIZE 1024
__global__ void
softmax_kernel(const bench_t *A, bench_t *B, bench_t *sum_d_B,const int size)
{   unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;
    bench_t value = 0;
    __shared__ bench_t shared_data[BLOCK_SIZE];
    if (i  < (size * size)){
        
        #ifdef INT
        value = exp(A[i]);
        #elif FLOAT
        value = expf(A[i]);
        #else
        value = exp(A[i]);
        #endif
        shared_data[tid] = value;
        B[i] = value;
        // sinc theads
        __syncthreads();
        for (unsigned int s=blockDim.x/2; s>0; s>>=1) 
        {
            if (tid < s)  
            {
                shared_data[tid] += shared_data[tid + s];
            }
        __syncthreads();
        }
        if (tid == 0){
            atomicAdd(sum_d_B, shared_data[0]);
        }    
    }
}
__global__ void
softmax_finish_kernel(bench_t *B, bench_t *sum_d_B,const int size)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i  < (size * size)){
        B[i] = (B[i]/(*sum_d_B));
    }
}

void init(GraficObject *device_object, char* device_name){
    init(device_object, 0,0, device_name);
}

void init(GraficObject *device_object, int platform ,int device, char* device_name){
    cudaSetDevice(device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    //printf("Using device: %s\n", prop.name);
    strcpy(device_name,prop.name);
    //event create 
    device_object->start = new cudaEvent_t;
    device_object->stop = new cudaEvent_t;
    device_object->start_memory_copy_device = new cudaEvent_t;
    device_object->stop_memory_copy_device = new cudaEvent_t;
    device_object->start_memory_copy_host = new cudaEvent_t;
    device_object->stop_memory_copy_host= new cudaEvent_t;
    
    cudaEventCreate(device_object->start);
    cudaEventCreate(device_object->stop);
    cudaEventCreate(device_object->start_memory_copy_device);
    cudaEventCreate(device_object->stop_memory_copy_device);
    cudaEventCreate(device_object->start_memory_copy_host);
    cudaEventCreate(device_object->stop_memory_copy_host);
}


bool device_memory_init(GraficObject *device_object, unsigned int size_a_matrix, unsigned int size_b_matrix){

   // Allocate the device input vector A
    cudaError_t err = cudaSuccess;
    err = cudaMalloc((void **)&device_object->d_A, size_a_matrix * sizeof(bench_t));

    if (err != cudaSuccess)
    {
        return false;
    }

    // Allocate the device input vector B
    err = cudaMalloc((void **)&device_object->d_B, size_b_matrix * sizeof(bench_t));

    if (err != cudaSuccess)
    {
        return false;
    }
    // Allocate the device input sum_d_B
    err = cudaMalloc((void **)&device_object->sum_d_B, sizeof(bench_t) );

    if (err != cudaSuccess)
    {
        return false;
    }


    return true;

}

void copy_memory_to_device(GraficObject *device_object, bench_t* h_A, unsigned int size_a){
    cudaEventRecord(*device_object->start_memory_copy_device);
    cudaError_t err = cudaMemcpy(device_object->d_A, h_A, sizeof(bench_t) * size_a, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    cudaEventRecord(*device_object->stop_memory_copy_device);   
}
void execute_kernel(GraficObject *device_object, unsigned int n, unsigned int m,unsigned int w){
    dim3 dimBlock, dimGrid;

    dimBlock = dim3(BLOCK_SIZE);
    dimGrid = dim3(ceil(float((n*n))/(dimBlock.x)));
    
    
    cudaEventRecord(*device_object->start);
    softmax_kernel<<<dimGrid, dimBlock>>>(device_object->d_A, device_object->d_B, device_object->sum_d_B, n);
    softmax_finish_kernel<<<dimGrid, dimBlock>>>(device_object->d_B, device_object->sum_d_B, n);
    cudaEventRecord(*device_object->stop);
}

void copy_memory_to_host(GraficObject *device_object, bench_t* h_C, int size){
    cudaEventRecord(*device_object->start_memory_copy_host);
    cudaMemcpy(h_C, device_object->d_B, size * sizeof(bench_t), cudaMemcpyDeviceToHost);
    cudaEventRecord(*device_object->stop_memory_copy_host);
    }

float get_elapsed_time(GraficObject *device_object, bool csv_format){
    cudaEventSynchronize(*device_object->stop_memory_copy_host);
    float milliseconds_h_d = 0, milliseconds = 0, milliseconds_d_h = 0;
    // memory transfer time host-device
    cudaEventElapsedTime(&milliseconds_h_d, *device_object->start_memory_copy_device, *device_object->stop_memory_copy_device);
    // kernel time
    cudaEventElapsedTime(&milliseconds, *device_object->start, *device_object->stop);
    //  memory transfer time device-host
    cudaEventElapsedTime(&milliseconds_d_h, *device_object->start_memory_copy_host, *device_object->stop_memory_copy_host);
    
    if (csv_format){
         printf("%.10f;%.10f;%.10f;\n", milliseconds_h_d,milliseconds,milliseconds_d_h);
    }else{
         printf("Elapsed time Host->Device: %.10f miliseconds\n", milliseconds_h_d);
         printf("Elapsed time kernel: %.10f miliseconds\n", milliseconds);
         printf("Elapsed time Device->Host: %.10f miliseconds\n", milliseconds_d_h);
    }
    return milliseconds;
}

void clean(GraficObject *device_object){
    cudaError_t err = cudaSuccess;
    err = cudaFree(device_object->d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaFree(device_object->d_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaFree(device_object->sum_d_B);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device  sum_d_B (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    // delete events
    delete device_object->start;
    delete device_object->stop;
    delete device_object->start_memory_copy_device;
    delete device_object->stop_memory_copy_device;
    delete device_object->start_memory_copy_host;
    delete device_object->stop_memory_copy_host;
}
