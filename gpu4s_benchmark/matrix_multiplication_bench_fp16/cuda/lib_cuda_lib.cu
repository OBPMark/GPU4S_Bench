#include <cublas_v2.h>
#include "../benchmark_library.h"

#define BLOCK_SIZE 16

__global__ void convert_fp32_to_f16 (bench_t *in, bench_t_gpu *out, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
       out[idx] = in[idx];
    }
 }


 __global__ void convert_fp16_to_f32 (bench_t_gpu *in, bench_t *out, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
       out[idx] = in[idx];
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


bool device_memory_init(GraficObject *device_object, unsigned int size_a_matrix, unsigned int size_b_matrix, unsigned int size_c_matrix){

   // Allocate the device input vector A
	cudaError_t err = cudaSuccess;
    err = cudaMalloc((void **)&device_object->d_A, size_a_matrix * sizeof(bench_t_gpu));

    if (err != cudaSuccess)
    {
        return false;
    }

    // Allocate the device input vector B
    err = cudaMalloc((void **)&device_object->d_B, size_b_matrix * sizeof(bench_t_gpu));

    if (err != cudaSuccess)
    {
        return false;
    }
    // Allocate the device input vector A_half
    err = cudaMalloc((void **)&device_object->d_half_A, size_a_matrix * sizeof(bench_t_gpu));

    if (err != cudaSuccess)
    {
        return false;
    }

    // Allocate the device input vector A_half
    err = cudaMalloc((void **)&device_object->d_half_B, size_b_matrix * sizeof(bench_t_gpu));

    if (err != cudaSuccess)
    {
        return false;
    }
    // Allocate the device output vector C
    err = cudaMalloc((void **)&device_object->d_C, size_c_matrix * sizeof(bench_t));

    if (err != cudaSuccess)
    {
        return false;
    }
    err = cudaMalloc((void **)&device_object->d_half_C, size_c_matrix * sizeof(bench_t_gpu));

    if (err != cudaSuccess)
    {
        return false;
    }
    return true;
}

void copy_memory_to_device(GraficObject *device_object, bench_t* h_A, bench_t* h_B, unsigned int size_a, unsigned int size_b){
    cudaEventRecord(*device_object->start_memory_copy_device);
	cudaError_t err = cudaMemcpy(device_object->d_A, h_A, sizeof(bench_t_gpu) * size_a, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    err = cudaMemcpy(device_object->d_B, h_B, sizeof(bench_t) * size_b, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    #ifdef FLOAT16
    // transform to half
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(ceil(float((size_a))/(dimBlock.x)));
    convert_fp32_to_f16<<<dimGrid, dimBlock>>> (device_object->d_A, device_object->d_half_A,size_a);
    dimGrid.x = ceil(float((size_b))/(dimBlock.x));
    convert_fp32_to_f16<<<dimGrid, dimBlock>>> (device_object->d_B, device_object->d_half_B,size_b);
    #endif
    cudaEventRecord(*device_object->stop_memory_copy_device);   
}
void execute_kernel(GraficObject *device_object, unsigned int n, unsigned int m,unsigned int w){
    // cublas settings
    int lda=m,ldb=m,ldc=m;
    const __half alf = 1;
    const __half bet = 0;
    const __half *alpha = &alf;
    const __half *beta = &bet;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaEventRecord(*device_object->start);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    #ifdef INT
    printf("CUBLAS NOT SUPPORT INT OPERATIOS\n");
    #elif FLOAT                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, w, alpha, device_object->d_half_B, lda, device_object->d_half_A, ldb, beta, device_object->d_half_C, ldc);
    #else 
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, w, alpha, device_object->d_B, lda, device_object->d_A, ldb, beta, device_object->d_C, ldc);
    #endif
    
    cudaEventRecord(*device_object->stop);
    // destroy cublas
    cublasDestroy(handle);
}

void copy_memory_to_host(GraficObject *device_object, bench_t* h_C, int size){

    cudaEventRecord(*device_object->start_memory_copy_host);
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(ceil(float((size))/(dimBlock.x)));
    convert_fp16_to_f32<<<dimGrid, dimBlock>>> (device_object->d_half_C, device_object->d_C,size);
    cudaMemcpy(h_C, device_object->d_C, size * sizeof(bench_t_gpu), cudaMemcpyDeviceToHost);
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
         printf("Elapsed time Host->Device: %.10f milliseconds\n", milliseconds_h_d);
         printf("Elapsed time kernel: %.10f milliseconds\n", milliseconds);
         printf("Elapsed time Device->Host: %.10f milliseconds\n", milliseconds_d_h);
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
    err = cudaFree(device_object->d_C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
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
