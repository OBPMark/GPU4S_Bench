/** * ====================================================================
 * @file        hip_common.cpp (./LRN_bench)
 * @brief       Common HIP platform initialization, device setup, 
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
    (void)hipSetDevice(device);
    hipDeviceProp_t prop;
    (void)hipGetDeviceProperties(&prop, device);
    //printf("Using device: %s\n", prop.name);
    strcpy(device_name,prop.name);
    //event create 
    deviceObj->start = new hipEvent_t;
    deviceObj->stop = new hipEvent_t;
    deviceObj->start_memory_copy_device = new hipEvent_t;
    deviceObj->stop_memory_copy_device = new hipEvent_t;
    deviceObj->start_memory_copy_host = new hipEvent_t;
    deviceObj->stop_memory_copy_host= new hipEvent_t;
    
    (void)hipEventCreate(deviceObj->start);
    (void)hipEventCreate(deviceObj->stop);
    (void)hipEventCreate(deviceObj->start_memory_copy_device);
    (void)hipEventCreate(deviceObj->stop_memory_copy_device);
    (void)hipEventCreate(deviceObj->start_memory_copy_host);
    (void)hipEventCreate(deviceObj->stop_memory_copy_host);
}

bool device_memory_init(GraficCommon* device_object, unsigned int size_a_matrix, unsigned int size_b_matrix){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);

    // FIX: create a dumb obj to sync the profling clock
    if (deviceObj->profiling_clock)
    {
       hipDumbSync();
    }
    
    // Allocate the device input vector A
    hipError_t  err = hipMalloc((void **)&deviceObj->d_A, size_a_matrix * sizeof(bench_t));
    if (err != hipSuccess) return false;

    // Allocate the device input vector B
    err = hipMalloc((void **)&deviceObj->d_B, size_b_matrix * sizeof(bench_t));
    if (err != hipSuccess) return false;

    return true;
}

void copy_memory_to_device(GraficCommon* device_object, bench_t* h_A, unsigned int size_a){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    // host -> device 
    Clock h2dCLK;

    // profilling start 
    h2dCLK.start();
    (void)hipEventRecord(*deviceObj->start_memory_copy_device);

    hipError_t err = hipMemcpy(deviceObj->d_A, h_A, sizeof(bench_t) * size_a, hipMemcpyHostToDevice);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", hipGetErrorString(err));
        return;
    }

    // profilling end
    (void)hipEventRecord(*deviceObj->stop_memory_copy_device);
    h2dCLK.end();

    // store the h2d time
    deviceObj->h2d_elapsed_time = h2dCLK.getElapsedMS();
}


void copy_memory_to_host(GraficCommon* device_object, bench_t* h_C, int size){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    // device ->  host
    Clock d2hCLK;

    // profilling start 
    d2hCLK.start();
    (void)hipEventRecord(*deviceObj->start_memory_copy_host);

    hipError_t err = hipMemcpy(h_C, deviceObj->d_B, size * sizeof(bench_t), hipMemcpyDeviceToHost);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from device to host (error code %s)!\n", hipGetErrorString(err));
        return;
    }
    
    // profilling end 
    (void)hipEventRecord(*deviceObj->stop_memory_copy_host);
    d2hCLK.end();
    
    // store the hd2h time
    deviceObj->d2h_elapsed_time = d2hCLK.getElapsedMS();
}

float get_elapsed_time(GraficCommon* device_object, bool csv_format, bool csv_format_timestamp, long int current_time){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    (void)hipEventSynchronize(*deviceObj->stop_memory_copy_host); // wait

    float milliseconds_h_d = 0, milliseconds = 0, milliseconds_d_h = 0;
    
    if (deviceObj->profiling_clock)
    {
        // --- FIX: Use <chrono> instead of CLBlast event profiling (unreliable on PROFILING_CLOCK) ---
        milliseconds_h_d  = deviceObj->h2d_elapsed_time;
        milliseconds      = deviceObj->elapsed_time;
        milliseconds_d_h  = deviceObj->d2h_elapsed_time;
    }else{
        // memory transfer time host-device
        (void)hipEventElapsedTime(&milliseconds_h_d, *deviceObj->start_memory_copy_device, *deviceObj->stop_memory_copy_device);
        // kernel time
        (void)hipEventElapsedTime(&milliseconds, *deviceObj->start, *deviceObj->stop);
        //  memory transfer time device-host
        (void)hipEventElapsedTime(&milliseconds_d_h, *deviceObj->start_memory_copy_host, *deviceObj->stop_memory_copy_host);
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

    hipError_t err = hipFree(deviceObj->d_A);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", hipGetErrorString(err));
        return;
    }

    err = hipFree(deviceObj->d_B);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", hipGetErrorString(err));
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