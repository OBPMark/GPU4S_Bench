/** * ====================================================================
 * @file        cuda_common.cu (./convolution_2D_bench)
 * @brief       Common CUDA platform initialization, device setup, 
 *              profiling timer evaluation, and generic cleanup routines.
 * @paragraph   License
 * ESA-PL Strong Copyleft – v2.5
 * ======================================================================= */
#include "../benchmark_library.h"

void init(GraficCommon* device_object, char* device_name){
	init(device_object, 0,0, device_name);
}

void init(GraficCommon* device_object, int platform ,int device, char* device_name){
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	cudaSetDevice(device);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);
	//printf("Using device: %s\n", prop.name);
    strcpy(device_name,prop.name);
    //event create 
    deviceObj->start = new cudaEvent_t;
    deviceObj->stop = new cudaEvent_t;
    deviceObj->start_memory_copy_device = new cudaEvent_t;
    deviceObj->stop_memory_copy_device = new cudaEvent_t;
    deviceObj->start_memory_copy_host = new cudaEvent_t;
    deviceObj->stop_memory_copy_host= new cudaEvent_t;
    
    cudaEventCreate(deviceObj->start);
    cudaEventCreate(deviceObj->stop);
    cudaEventCreate(deviceObj->start_memory_copy_device);
    cudaEventCreate(deviceObj->stop_memory_copy_device);
    cudaEventCreate(deviceObj->start_memory_copy_host);
    cudaEventCreate(deviceObj->stop_memory_copy_host);
}

bool device_memory_init(GraficCommon* device_object, unsigned int size_a_matrix, unsigned int size_b_matrix, unsigned int size_c_matrix){
   
GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
   
   // Allocate the device input vector A
    cudaError_t err = cudaMalloc((void **)&deviceObj->d_A, size_a_matrix * sizeof(bench_t));
    if (err != cudaSuccess) return false;

    // Allocate the device input vector B
    err = cudaMalloc((void **)&deviceObj->d_B, size_b_matrix * sizeof(bench_t));
    if (err != cudaSuccess) return false;

    // Allocate the device output vector C
    err = cudaMalloc((void **)&deviceObj->kernel, size_c_matrix * sizeof(bench_t));
    if (err != cudaSuccess) return false;

    return true;
}

void copy_memory_to_device(GraficCommon* device_object, bench_t* h_A, bench_t* kernel, unsigned int size_a, unsigned int size_b){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    // host -> device 
    Clock h2dCLK;

    // profilling start 
    h2dCLK.start();
    cudaEventRecord(*deviceObj->start_memory_copy_device);

	cudaError_t err = cudaMemcpy(deviceObj->d_A, h_A, sizeof(bench_t) * size_a, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    err = cudaMemcpy(deviceObj->kernel, kernel, sizeof(bench_t) * size_b, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector kernel from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    // profilling end
    cudaEventRecord(*deviceObj->stop_memory_copy_device);
    h2dCLK.end();
    
    // store the h2d time
    deviceObj->h2d_elapsed_time = h2dCLK.getElapsedMS();
    
}



void copy_memory_to_host(GraficCommon* device_object, bench_t* h_C, int size){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    // host -> device 
    Clock d2hCLK;

    // profilling start 
    d2hCLK.start();
    cudaEventRecord(*deviceObj->start_memory_copy_host);

    cudaError_t err = cudaMemcpy(h_C, deviceObj->d_B, size * sizeof(bench_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from device to host (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    
    // profilling end
    cudaEventRecord(*deviceObj->stop_memory_copy_host);
    d2hCLK.end();

    // store the hd2h time
    deviceObj->d2h_elapsed_time = d2hCLK.getElapsedMS();
}

float get_elapsed_time(GraficCommon* device_object, bool csv_format,bool csv_format_timestamp, long int current_time){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    cudaEventSynchronize(*deviceObj->stop_memory_copy_host); // wait
    float milliseconds_h_d = 0, milliseconds = 0, milliseconds_d_h = 0;
    
    if (deviceObj->profiling_clock)
    {
        // --- FIX: Use <chrono> instead of CLBlast event profiling (unreliable on PROFILING_CLOCK) ---
        milliseconds_h_d  = deviceObj->h2d_elapsed_time;
        milliseconds      = deviceObj->elapsed_time;
        milliseconds_d_h  = deviceObj->d2h_elapsed_time;
    }else{
        // memory transfer time host-device
        cudaEventElapsedTime(&milliseconds_h_d, *deviceObj->start_memory_copy_device, *deviceObj->stop_memory_copy_device);
        // kernel time
        cudaEventElapsedTime(&milliseconds, *deviceObj->start, *deviceObj->stop);
        //  memory transfer time device-host
        cudaEventElapsedTime(&milliseconds_d_h, *deviceObj->start_memory_copy_host, *deviceObj->stop_memory_copy_host);
    }
    
    
    if (csv_format_timestamp){
        printf("%.10f;%.10f;%.10f;%ld;\n", milliseconds_h_d,milliseconds,milliseconds_d_h, current_time);
    }
    else if (csv_format){
         printf("%.10f;%.10f;%.10f;\n", milliseconds_h_d,milliseconds,milliseconds_d_h);
    }else{
         printf("profiling mode: %s\n", deviceObj->profiling_clock ? "CLOCK" : "GPU");
         printf("Elapsed time Host->Device: %.10f milliseconds\n", milliseconds_h_d);
         printf("Elapsed time kernel: %.10f milliseconds\n", milliseconds);
         printf("Elapsed time Device->Host: %.10f milliseconds\n", milliseconds_d_h);
    }
    return milliseconds;
}

void clean(GraficCommon* device_object){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);

    cudaError_t err = cudaFree(deviceObj->d_A);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaFree(deviceObj->d_B);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaFree(deviceObj->kernel);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    // delete events
    delete deviceObj->start;
    delete deviceObj->stop;
    delete deviceObj->start_memory_copy_device;
    delete deviceObj->stop_memory_copy_device;
    delete deviceObj->start_memory_copy_host;
    delete deviceObj->stop_memory_copy_host;
}
