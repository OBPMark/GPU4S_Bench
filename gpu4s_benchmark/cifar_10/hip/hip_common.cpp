/** * ====================================================================
 * @file        hip_common.cpp (./cifar_10)
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

bool device_memory_init(GraficCommon* device_object, unsigned int input_data, unsigned int output_data, unsigned int kernel_1, unsigned int kernel_2, unsigned int stride_1, unsigned int stride_2, unsigned int neurons_dense_1, unsigned int neurons_dense_2){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    
    // FIX: create a dumb obj to sync the profling clock
    if (deviceObj->profiling_clock)
    {
       hipDumbSync();
    }
        // Allocate input
    hipError_t err = hipMalloc((void **)&deviceObj->input_data, input_data * input_data * sizeof(bench_t));
    if (err != hipSuccess) return false;

    // Allocate kernel
    err = hipMalloc((void **)&deviceObj->kernel_1, kernel_1 * kernel_1 * sizeof(bench_t));
    if (err != hipSuccess) return false;

    // Allocate conv 1 output
    err = hipMalloc((void **)&deviceObj->conv_1_output, input_data * input_data * sizeof(bench_t));
    if (err != hipSuccess) return false;

    // Allocate pooling output
    unsigned int size_pooling_1 = input_data / stride_1;
    err = hipMalloc((void **)&deviceObj->pooling_1_output, size_pooling_1 * size_pooling_1 * sizeof(bench_t));
    if (err != hipSuccess) return false;

    // Allocate kernel 2
    err = hipMalloc((void **)&deviceObj->kernel_2, kernel_2 * kernel_2 * sizeof(bench_t));
    if (err != hipSuccess) return false;

    // Allocate conv 1 output
    err = hipMalloc((void **)&deviceObj->conv_2_output, size_pooling_1 * size_pooling_1 * sizeof(bench_t));
    if (err != hipSuccess) return false;

    // Allocate pooling output
    unsigned int size_pooling_2 = size_pooling_1 / stride_2;
    if (err != hipSuccess) return false;

    //dense layer 1 weights 
    unsigned int weights_layer_1 = size_pooling_2 * size_pooling_2 * neurons_dense_1;

    err = hipMalloc((void **)&deviceObj->dense_layer_1_weights, weights_layer_1* sizeof(bench_t));
    if (err != hipSuccess) return false;
    
    // dense layer output 1
    err = hipMalloc((void **)&deviceObj->dense_layer_1_output, neurons_dense_1 * sizeof(bench_t));
    if (err != hipSuccess) return false;

    //dense layer 2 weights 
    unsigned int weights_layer_2 = neurons_dense_1 * neurons_dense_2;
    err = hipMalloc((void **)&deviceObj->dense_layer_2_weights, weights_layer_2  * sizeof(bench_t));
    if (err != hipSuccess) return false;

    // dense layer output 2
    err = hipMalloc((void **)&deviceObj->dense_layer_2_output, neurons_dense_2 * sizeof(bench_t));
    if (err != hipSuccess) return false;

     // sum data
    err = hipMalloc((void **)&deviceObj->sum_ouput, sizeof(bench_t));
    if (err != hipSuccess) return false;

    // output data
    err = hipMalloc((void **)&deviceObj->output_data, neurons_dense_2 * sizeof(bench_t));
    if (err != hipSuccess) return false;

    return true;
 }

void copy_memory_to_device(GraficCommon* device_object, bench_t* input_data, bench_t* kernel_1_data, bench_t* kernel_2_data, bench_t* weights_1 ,bench_t* weights_2,unsigned int input , unsigned int kernel_size_1, unsigned int kernel_size_2, unsigned int weights_1_size, unsigned int weights_2_size){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    // host -> device 
    Clock h2dCLK;

    // profilling start 
    h2dCLK.start();
    (void)hipEventRecord(*deviceObj->start_memory_copy_device);

    hipError_t err = hipMemcpy(deviceObj->input_data, input_data, sizeof(bench_t) * input * input, hipMemcpyHostToDevice);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to copy vector input from host to device (error code %s)!\n", hipGetErrorString(err));
        return;
    }
    err = hipMemcpy(deviceObj->kernel_1, kernel_1_data, sizeof(bench_t) * kernel_size_1 * kernel_size_1, hipMemcpyHostToDevice);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to copy vector kernel_1 from host to device (error code %s)!\n", hipGetErrorString(err));
        return;
    }
    err = hipMemcpy(deviceObj->kernel_2, kernel_2_data, sizeof(bench_t) * kernel_size_2 * kernel_size_2, hipMemcpyHostToDevice);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to copy vector kernel_2 from host to device (error code %s)!\n", hipGetErrorString(err));
        return;
    }
    err = hipMemcpy(deviceObj->dense_layer_1_weights, weights_1, sizeof(bench_t) * weights_1_size, hipMemcpyHostToDevice);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to copy vector weights_layer_1 from host to device (error code %s)!\n", hipGetErrorString(err));
        return;
    }
    err = hipMemcpy(deviceObj->dense_layer_2_weights, weights_2, sizeof(bench_t) * weights_2_size, hipMemcpyHostToDevice);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to copy vector weights_layer_2 from host to device (error code %s)!\n", hipGetErrorString(err));
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
    
    hipError_t err = hipMemcpy(h_C, deviceObj->output_data, size * sizeof(bench_t), hipMemcpyDeviceToHost);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to copy vector output_data from device to host (error code %s)!\n", hipGetErrorString(err));
        return;
    }
    //hipMemcpy(h_C, deviceObj->dense_layer_2_output, 10 * sizeof(bench_t), hipMemcpyDeviceToHost);
    
    // profilling end 
    (void)hipEventRecord(*deviceObj->stop_memory_copy_host);
    d2hCLK.end();
    
    // store the hd2h time
    deviceObj->d2h_elapsed_time = d2hCLK.getElapsedMS();
}

float get_elapsed_time(GraficCommon* device_object, bool csv_format,bool csv_format_timestamp, long int current_time){
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
        printf("%.10f;%.10f;%.10f;%ld;\n",  milliseconds_h_d,milliseconds,milliseconds_d_h, current_time);
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

    hipError_t err = hipFree(deviceObj->input_data);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to free device vector input_data (error code %s)!\n", hipGetErrorString(err));
        return;
    }

    err = hipFree(deviceObj->kernel_1);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to free device vector kernel_1 (error code %s)!\n", hipGetErrorString(err));
        return;
    }

    err = hipFree(deviceObj->conv_1_output);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to free device vector conv_1_output (error code %s)!\n", hipGetErrorString(err));
        return;
    }

    err = hipFree(deviceObj->pooling_1_output);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to free device vector pooling_1_output (error code %s)!\n", hipGetErrorString(err));
        return;
    }

    err = hipFree(deviceObj->kernel_2);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to free device vector kernel_2 (error code %s)!\n", hipGetErrorString(err));
        return;
    }

    err = hipFree(deviceObj->conv_2_output);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to free device vector conv_2_output (error code %s)!\n", hipGetErrorString(err));
        return;
    }

    err = hipFree(deviceObj->pooling_2_output);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to free device vector pooling_2_output (error code %s)!\n", hipGetErrorString(err));
        return;
    }

    err = hipFree(deviceObj->dense_layer_1_weights);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to free device vector dense_layer_1_weights (error code %s)!\n", hipGetErrorString(err));
        return;
    }

    err = hipFree(deviceObj->dense_layer_2_weights);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to free device vector dense_layer_2_weights (error code %s)!\n", hipGetErrorString(err));
        return;
    }

    err = hipFree(deviceObj->dense_layer_1_output);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to free device vector dense_layer_1_output (error code %s)!\n", hipGetErrorString(err));
        return;
    }

    err = hipFree(deviceObj->dense_layer_2_output);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to free device vector dense_layer_2_output (error code %s)!\n", hipGetErrorString(err));
        return;
    }

    err = hipFree(deviceObj->output_data);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to free device vector output_data (error code %s)!\n", hipGetErrorString(err));
        return;
    }

    err = hipFree(deviceObj->sum_ouput);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to free device vector sum_ouput (error code %s)!\n", hipGetErrorString(err));
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
