#include "hip/hip_runtime.h"
#include "../benchmark_library.h"

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
//#define BLOCK_SIZE 16
__global__ void
covolution_kernel(const bench_t *A, bench_t *B, const bench_t *kernel,const int n, const int m, const int w, const int kernel_size, const int shared_size, const int kernel_rad)
{
    unsigned int size = n;
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x0, y0;
    HIP_DYNAMIC_SHARED( bench_t, data)
    if (x < size && y < size)
    {
        // each thread load 4 values ,the corners
        //TOP right corner
        x0 = x - kernel_rad;
        y0 = y - kernel_rad;
        //printf("POS x %d y %d x0 %d y0 %d\n", threadIdx.x, threadIdx.y, x0, y0);
        if ( x0 < 0 || y0 < 0 )
        {
            data[threadIdx.x * shared_size + threadIdx.y] = 0;
        }
        else
        {
            data[threadIdx.x * shared_size + threadIdx.y] = A[x0 *size+y0];
        } 
            
        //BOTTOM right corner
        x0 = x + kernel_rad;
        y0 = y - kernel_rad;
        //printf("POS x %d y %d x0 %d y0 %d\n", threadIdx.x, threadIdx.y, x0, y0);
        if ( x0 > size-1  || y0 < 0 )
        {
            data[(threadIdx.x + kernel_rad * 2) * shared_size + threadIdx.y] = 0;
        }
        else
        {
            data[(threadIdx.x + kernel_rad * 2) * shared_size + threadIdx.y] = A[x0 *size+y0];
        } 

        //TOP left corner
        x0 = x - kernel_rad;
        y0 = y + kernel_rad;
        //printf("POS x %d y %d x0 %d y0 %d\n", threadIdx.x, threadIdx.y, x0, y0);
        if ( x0 < 0  || y0 > size-1 )
        {
            data[threadIdx.x * shared_size + (threadIdx.y + kernel_rad * 2)] = 0;
        }
        else
        {
            data[threadIdx.x * shared_size + (threadIdx.y + kernel_rad * 2)] = A[x0 *size+y0];
        } 

        //BOTTOM left corner
        x0 = x + kernel_rad;
        y0 = y + kernel_rad;
        //printf("POS x %d y %d x0 %d y0 %d\n", threadIdx.x, threadIdx.y, x0, y0);
        if ( x0 > size-1  || y0 > size-1 )
        {
            data[(threadIdx.x + kernel_rad * 2) * shared_size + (threadIdx.y + kernel_rad * 2)] = 0;
        }
        else
        {
            data[(threadIdx.x + kernel_rad * 2) * shared_size + (threadIdx.y + kernel_rad * 2)] = A[x0 *size+y0];
        } 

        __syncthreads();
        bench_t sum = 0;
        unsigned int xa = kernel_rad + threadIdx.x;
        unsigned int ya = kernel_rad + threadIdx.y;
        #pragma unroll
        for(int i = -kernel_rad; i <= kernel_rad; ++i) // loop over kernel_rad  -1 to 1 in kernel_size 3 
            {
                #pragma unroll
                for(int j = -kernel_rad; j <= kernel_rad; ++j)
                {
                    //printf("ACHIVED position  %d %d value %f\n", (xa + i) , (ya + j), data[(xa + i)][(ya + j)]);
                    sum += data[(xa + i) * shared_size +  (ya + j)] * kernel[(i+kernel_rad)* kernel_size + (j+kernel_rad)];
                }
            }
                
        B[x*size+y ] = sum;
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


bool device_memory_init(GraficObject *device_object, unsigned int size_a_matrix, unsigned int size_b_matrix, unsigned int size_c_matrix){
   
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

    // Allocate the device output vector C
    err = hipMalloc((void **)&device_object->kernel, size_c_matrix * sizeof(bench_t));

    if (err != hipSuccess)
    {
        return false;
    }
    return true;
}

void copy_memory_to_device(GraficObject *device_object, bench_t* h_A, bench_t* kernel, unsigned int size_a, unsigned int size_b){
    hipEventRecord(*device_object->start_memory_copy_device);
    hipError_t err = hipMemcpy(device_object->d_A, h_A, sizeof(bench_t) * size_a, hipMemcpyHostToDevice);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", hipGetErrorString(err));
        return;
    }
    err = hipMemcpy(device_object->kernel, kernel, sizeof(bench_t) * size_b, hipMemcpyHostToDevice);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to copy vector kernel from host to device (error code %s)!\n", hipGetErrorString(err));
        return;
    }
    hipEventRecord(*device_object->stop_memory_copy_device);
    
}
void execute_kernel(GraficObject *device_object, unsigned int n, unsigned int m,unsigned int w, unsigned int kernel_size){
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(ceil(float(n)/dimBlock.x), ceil(float(m)/dimBlock.y));
    unsigned int kernel_rad =  kernel_size / 2;
    unsigned int size_shared = (BLOCK_SIZE + kernel_rad *2 ) * sizeof(bench_t) * (BLOCK_SIZE + kernel_rad *2) * sizeof(bench_t);
    unsigned int size_shared_position = (BLOCK_SIZE + kernel_rad *2);
    hipEventRecord(*device_object->start);
    hipLaunchKernelGGL((covolution_kernel), dim3(dimGrid), dim3(dimBlock), size_shared , 0, device_object->d_A, device_object->d_B, device_object->kernel, n, m, w, kernel_size, size_shared_position, kernel_rad);
    hipEventRecord(*device_object->stop);
}

void copy_memory_to_host(GraficObject *device_object, bench_t* h_C, int size){
    hipEventRecord(*device_object->start_memory_copy_host);
    hipMemcpy(h_C, device_object->d_B, size * sizeof(bench_t), hipMemcpyDeviceToHost);
    hipEventRecord(*device_object->stop_memory_copy_host);
}

float get_elapsed_time(GraficObject *device_object, bool csv_format,bool csv_format_timestamp, long int current_time){
    hipEventSynchronize(*device_object->stop_memory_copy_host);
    float milliseconds_h_d = 0, milliseconds = 0, milliseconds_d_h = 0;
    // memory transfer time host-device
    hipEventElapsedTime(&milliseconds_h_d, *device_object->start_memory_copy_device, *device_object->stop_memory_copy_device);
    // kernel time
    hipEventElapsedTime(&milliseconds, *device_object->start, *device_object->stop);
    //  memory transfer time device-host
    hipEventElapsedTime(&milliseconds_d_h, *device_object->start_memory_copy_host, *device_object->stop_memory_copy_host);
    
    if (csv_format_timestamp){
        printf("%.10f;%.10f;%.10f;%ld;\n", milliseconds_h_d,milliseconds,milliseconds_d_h, current_time);
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
    err = hipFree(device_object->kernel);

    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", hipGetErrorString(err));
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