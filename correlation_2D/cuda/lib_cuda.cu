#include "../benchmark_library.h"

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
//#define BLOCK_SIZE 32
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

    // Allocate the device output R value
    err = cudaMalloc((void **)&device_object->d_R, sizeof(result_bench_t));

    if (err != cudaSuccess)
    {
        return false;
    }

    // Allocate the auxiliar values for matrix A and B

    err =  cudaMalloc((void **)&device_object->mean_A, sizeof(result_bench_t)); 
    if (err != cudaSuccess)
    {
        return false;
    }

    err =  cudaMalloc((void **)&device_object->mean_B, sizeof(result_bench_t)); 
    if (err != cudaSuccess)
    {
        return false;
    }

    err =  cudaMalloc((void **)&device_object->acumulate_value_a_b, sizeof(result_bench_t)); 
    if (err != cudaSuccess)
    {
        return false;
    }

    err =  cudaMalloc((void **)&device_object->acumulate_value_a_a, sizeof(result_bench_t)); 
    if (err != cudaSuccess)
    {
        return false;
    }
    err =  cudaMalloc((void **)&device_object->acumulate_value_b_b, sizeof(result_bench_t)); 
    if (err != cudaSuccess)
    {
        return false;
    }

    return true;
}

void copy_memory_to_device(GraficObject *device_object, bench_t* h_A, unsigned int size_a, bench_t* h_B, unsigned int size_b){
    cudaEventRecord(*device_object->start_memory_copy_device);
	cudaError_t err = cudaMemcpy(device_object->d_A, h_A, sizeof(bench_t) * size_a, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaMemcpy(device_object->d_B, h_B, sizeof(bench_t) * size_b, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    
    cudaEventRecord(*device_object->stop_memory_copy_device);
    
}
void execute_kernel(GraficObject *device_object, unsigned int n){
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    dim3 dimGrid(ceil(float(n)/dimBlock.x),ceil(float(n)/dimBlock.y));
    cudaEventRecord(*device_object->start);
    mean_matrices<<<dimGrid,dimBlock>>>(device_object->d_A, device_object->d_B, device_object->mean_A, device_object->mean_B , n);
    correlation_2D<<<dimGrid,dimBlock>>>(device_object->d_A, device_object->d_B, device_object->d_R, device_object->mean_A, device_object->mean_B,device_object->acumulate_value_a_b, device_object->acumulate_value_a_a, device_object->acumulate_value_b_b, n);

    cudaEventRecord(*device_object->stop);
}

void copy_memory_to_host(GraficObject *device_object, result_bench_t* h_R){
    cudaEventRecord(*device_object->start_memory_copy_host);
    result_bench_t acumulate_value_a_a;
    result_bench_t acumulate_value_a_b;
    result_bench_t acumulate_value_b_b;
    cudaMemcpy(&acumulate_value_a_a, device_object->acumulate_value_a_a, sizeof(result_bench_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&acumulate_value_a_b, device_object->acumulate_value_a_b, sizeof(result_bench_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&acumulate_value_b_b, device_object->acumulate_value_b_b, sizeof(result_bench_t), cudaMemcpyDeviceToHost);
    *h_R = (result_bench_t)(acumulate_value_a_b / (result_bench_t)(sqrt(acumulate_value_a_a * acumulate_value_b_b)));
    //cudaMemcpy(h_R, device_object->d_R, sizeof(result_bench_t), cudaMemcpyDeviceToHost);
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

    err = cudaFree(device_object->d_R);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device R (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    // delete auxiliars
    err = cudaFree(device_object->mean_A);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device  mean_A (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    err = cudaFree( device_object->mean_B);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device mean_B (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaFree(device_object->acumulate_value_a_b);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device acumulate_value_a_b (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    err = cudaFree(device_object->acumulate_value_a_a);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device acumulate_value_a_a (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    err = cudaFree(device_object->acumulate_value_b_b);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device acumulate_value_b_b (error code %s)!\n", cudaGetErrorString(err));
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
