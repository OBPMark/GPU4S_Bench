#include "hip/hip_runtime.h"
#include "../benchmark_library.h"
#include "math.h"

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
//#define BLOCK_SIZE 32
__global__ void
softmax_kernel(const bench_t *A, bench_t *B, bench_t *sum_d_B,const int size)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < size && j < size){
        #ifdef INT
        B[i*size+j] = exp(A[i*size+j]);
        #elif FLOAT
        B[i*size+j] = expf(A[i*size+j]);
        #else
        B[i*size+j] = exp(A[i*size+j]);
        #endif
        atomicAdd(sum_d_B, B[i*size+j]);
    }
}
__global__ void
softmax_finish_kernel(bench_t *B, bench_t *sum_d_B,const int size)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < size && j < size){
        B[i*size+j] = (B[i*size+j]/(*sum_d_B));
    }
}

void init(GraficObject *device_object, char* device_name){
    init(device_object, 0,0, device_name);
}

void init(GraficObject *device_object, int platform ,int device, char* device_name){
    hipSetDevice(device);
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, device);
    //printf("Using device: %s\n", prop.name);
    strcpy(device_name,prop.name);
    //event create 
    device_object->start = new hipEvent_t;
    device_object->stop = new hipEvent_t;
    device_object->start_memory_copy_device = new hipEvent_t;
    device_object->stop_memory_copy_device = new hipEvent_t;
    device_object->start_memory_copy_host = new hipEvent_t;
    device_object->stop_memory_copy_host= new hipEvent_t;
    
    hipEventCreate(device_object->start);
    hipEventCreate(device_object->stop);
    hipEventCreate(device_object->start_memory_copy_device);
    hipEventCreate(device_object->stop_memory_copy_device);
    hipEventCreate(device_object->start_memory_copy_host);
    hipEventCreate(device_object->stop_memory_copy_host);
}


bool device_memory_init(GraficObject *device_object, unsigned int size_a_matrix, unsigned int size_b_matrix){

   // Allocate the device input vector A
    hipError_t err = hipSuccess;
    err = hipMalloc((void **)&device_object->d_A, size_a_matrix * sizeof(bench_t));

    if (err != hipSuccess)
    {
        return false;
    }

    // Allocate the device input vector B
    err = hipMalloc((void **)&device_object->d_B, size_b_matrix * sizeof(bench_t));

    if (err != hipSuccess)
    {
        return false;
    }
    // Allocate the device input sum_d_B
    err = hipMalloc((void **)&device_object->sum_d_B, sizeof(bench_t));

    if (err != hipSuccess)
    {
        return false;
    }


    return true;

}

void copy_memory_to_device(GraficObject *device_object, bench_t* h_A, unsigned int size_a){
    hipEventRecord(*device_object->start_memory_copy_device);
    hipError_t err = hipMemcpy(device_object->d_A, h_A, sizeof(bench_t) * size_a, hipMemcpyHostToDevice);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", hipGetErrorString(err));
        return;
    }

    hipEventRecord(*device_object->stop_memory_copy_device);   
}
void execute_kernel(GraficObject *device_object, unsigned int n, unsigned int m,unsigned int w){
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(ceil(float(n)/dimBlock.x), ceil(float(m)/dimBlock.y));
    hipEventRecord(*device_object->start);
    hipLaunchKernelGGL((softmax_kernel), dim3(dimGrid), dim3(dimBlock), 0, 0, device_object->d_A, device_object->d_B, device_object->sum_d_B, n);
    hipLaunchKernelGGL((softmax_finish_kernel), dim3(dimGrid), dim3(dimBlock), 0, 0, device_object->d_B, device_object->sum_d_B, n);
    hipEventRecord(*device_object->stop);
}

void copy_memory_to_host(GraficObject *device_object, bench_t* h_C, int size){
    hipEventRecord(*device_object->start_memory_copy_host);
    hipMemcpy(h_C, device_object->d_B, size * sizeof(bench_t), hipMemcpyDeviceToHost);
    hipEventRecord(*device_object->stop_memory_copy_host);
    }

float get_elapsed_time(GraficObject *device_object, bool csv_format, bool csv_format_timestamp, long int current_time){
    hipEventSynchronize(*device_object->stop_memory_copy_host);
    float milliseconds_h_d = 0, milliseconds = 0, milliseconds_d_h = 0;
    // memory transfer time host-device
    hipEventElapsedTime(&milliseconds_h_d, *device_object->start_memory_copy_device, *device_object->stop_memory_copy_device);
    // kernel time
    hipEventElapsedTime(&milliseconds, *device_object->start, *device_object->stop);
    //  memory transfer time device-host
    hipEventElapsedTime(&milliseconds_d_h, *device_object->start_memory_copy_host, *device_object->stop_memory_copy_host);
    
    if (csv_format_timestamp){
        printf("%.10f;%.10f;%.10f;%ld;\n", milliseconds_h_d,milliseconds,milliseconds_d_h,current_time);
    }
    else if (csv_format){
         printf("%.10f;%.10f;%.10f;\n", milliseconds_h_d,milliseconds,milliseconds_d_h);
    }else{
         printf("Elapsed time Host->Device: %.10f milliseconds\n", milliseconds_h_d);
         printf("Elapsed time kernel: %.10f milliseconds\n", milliseconds);
         printf("Elapsed time Device->Host: %.10f milliseconds\n", milliseconds_d_h);
    }
    return milliseconds;
}

void clean(GraficObject *device_object){
    hipError_t err = hipSuccess;
    err = hipFree(device_object->d_A);

    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", hipGetErrorString(err));
        return;
    }

    err = hipFree(device_object->d_B);

    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", hipGetErrorString(err));
        return;
    }

    err = hipFree(device_object->sum_d_B);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to free device  sum_d_B (error code %s)!\n", hipGetErrorString(err));
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
