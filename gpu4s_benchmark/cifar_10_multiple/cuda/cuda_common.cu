/** * ====================================================================
 * @file        cuda_common.cu (./cifar_10_multiple)
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

bool device_memory_init(GraficCommon* device_object, unsigned int input_data, unsigned int output_data, unsigned int kernel_1, unsigned int kernel_2, unsigned int stride_1, unsigned int stride_2, unsigned int neurons_dense_1, unsigned int neurons_dense_2, unsigned int number_of_images){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    // Allocate input

    cudaError_t err = cudaMalloc((void **)&(deviceObj->input_data), number_of_images * input_data * input_data * sizeof(bench_t));
    if (err != cudaSuccess) return false;
    
    // Allocate kernel
    err = cudaMalloc((void **)&(deviceObj->kernel_1), kernel_1 * kernel_1 * sizeof(bench_t));
    if (err != cudaSuccess) return false;
    
    // Allocate conv 1 output
    err = cudaMalloc((void **)&(deviceObj->conv_1_output), input_data * input_data * sizeof(bench_t));
    if (err != cudaSuccess) return false;
    
    // Allocate pooling output
    unsigned int size_pooling_1 = input_data / stride_1;
    err = cudaMalloc((void **)&(deviceObj->pooling_1_output), size_pooling_1 * size_pooling_1 * sizeof(bench_t));
    if (err != cudaSuccess) return false;
    
    // Allocate kernel 2
    err = cudaMalloc((void **)&(deviceObj->kernel_2), kernel_2 * kernel_2 * sizeof(bench_t));
    if (err != cudaSuccess) return false;
    
    // Allocate conv 1 output
    err = cudaMalloc((void **)&(deviceObj->conv_2_output), size_pooling_1 * size_pooling_1 * sizeof(bench_t));
    if (err != cudaSuccess) return false;
   
    // Allocate pooling output
    unsigned int size_pooling_2 = size_pooling_1 / stride_2;
    err = cudaMalloc((void **)&(deviceObj->pooling_2_output), size_pooling_2 * size_pooling_2 * sizeof(bench_t));
    if (err != cudaSuccess) return false;
    
    //dense layer 1 weights 
    unsigned int weights_layer_1 = size_pooling_2 * size_pooling_2 * neurons_dense_1;
    err = cudaMalloc((void **)&(deviceObj->dense_layer_1_weights), weights_layer_1* sizeof(bench_t));
    if (err != cudaSuccess) return false;
   

    // dense layer output 1
    err = cudaMalloc((void **)&(deviceObj->dense_layer_1_output), neurons_dense_1 * sizeof(bench_t));
    if (err != cudaSuccess) return false;

    //dense layer 2 weights 
    unsigned int weights_layer_2 = neurons_dense_1 * neurons_dense_2;
    err = cudaMalloc((void **)&(deviceObj->dense_layer_2_weights), weights_layer_2  * sizeof(bench_t));
    if (err != cudaSuccess) return false;
  
    // dense layer output 2
    err = cudaMalloc((void **)&(deviceObj->dense_layer_2_output), neurons_dense_2 * sizeof(bench_t));
    if (err != cudaSuccess) return false;
    
     // sum data
    err = cudaMalloc((void **)&(deviceObj->sum_ouput), sizeof(bench_t));
    if (err != cudaSuccess) return false;
    
    // output data
    err = cudaMalloc((void **)&(deviceObj->output_data), number_of_images * neurons_dense_2 * sizeof(bench_t));
    if (err != cudaSuccess) return false;

    return true;
 }

void copy_memory_to_device(GraficCommon* device_object, bench_t* input_data, bench_t* kernel_1_data, bench_t* kernel_2_data, bench_t* weights_1 ,bench_t* weights_2,unsigned int input , unsigned int kernel_size_1, unsigned int kernel_size_2, unsigned int weights_1_size, unsigned int weights_2_size, unsigned int number_of_images){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    // host -> device 
    Clock h2dCLK;

    // profilling start 
    h2dCLK.start();
    cudaEventRecord(*deviceObj->start_memory_copy_device);

	cudaError_t err = cudaMemcpy(deviceObj->input_data, input_data, sizeof(bench_t) * input * input * number_of_images, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector input from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    err = cudaMemcpy(deviceObj->kernel_1, kernel_1_data, sizeof(bench_t) * kernel_size_1 * kernel_size_1, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector kernel_1 from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    err = cudaMemcpy(deviceObj->kernel_2, kernel_2_data, sizeof(bench_t) * kernel_size_2 * kernel_size_2, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector kernel_2 from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    err = cudaMemcpy(deviceObj->dense_layer_1_weights, weights_1, sizeof(bench_t) * weights_1_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector weights_layer_1 from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    err = cudaMemcpy(deviceObj->dense_layer_2_weights, weights_2, sizeof(bench_t) * weights_2_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector weights_layer_2 from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    cudaMemset(deviceObj->sum_ouput, 0,  sizeof(bench_t));

    // profilling end
    cudaEventRecord(*deviceObj->stop_memory_copy_device);
    h2dCLK.end();
    
    // store the h2d time
    deviceObj->h2d_elapsed_time = h2dCLK.getElapsedMS();
}

void copy_memory_to_host(GraficCommon* device_object, bench_t* h_C, int size, unsigned int number_of_images){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    // host -> device 
    Clock d2hCLK;

    // profilling start 
    d2hCLK.start();
    cudaEventRecord(*deviceObj->start_memory_copy_host);

    cudaError_t err = cudaMemcpy(h_C, deviceObj->output_data, number_of_images * size * sizeof(bench_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector output_data from device to host (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    //cudaMemcpy(h_C, deviceObj->dense_layer_2_output, 10 * sizeof(bench_t), cudaMemcpyDeviceToHost);
   
    // profilling end
    cudaEventRecord(*deviceObj->stop_memory_copy_host);
    d2hCLK.end();

    // store the hd2h time
    deviceObj->d2h_elapsed_time = d2hCLK.getElapsedMS();
}

float get_elapsed_time(GraficCommon* device_object, bool csv_format,bool csv_format_timestamp, long int current_time)
{
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    cudaEventSynchronize(*deviceObj->stop_memory_copy_host); //wait
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

    cudaError_t err = cudaFree(deviceObj->input_data);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector input_data (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaFree(deviceObj->kernel_1);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector kernel_1 (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaFree(deviceObj->conv_1_output);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector conv_1_output (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaFree(deviceObj->pooling_1_output);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector pooling_1_output (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaFree(deviceObj->kernel_2);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector kernel_2 (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaFree(deviceObj->conv_2_output);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector conv_2_output (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaFree(deviceObj->pooling_2_output);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector pooling_2_output (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaFree(deviceObj->dense_layer_1_weights);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector dense_layer_1_weights (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaFree(deviceObj->dense_layer_2_weights);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector dense_layer_2_weights (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaFree(deviceObj->dense_layer_1_output);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector dense_layer_1_output (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaFree(deviceObj->dense_layer_2_output);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector dense_layer_2_output (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaFree(deviceObj->output_data);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector output_data (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    
    err = cudaFree(deviceObj->sum_ouput);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector sum_ouput (error code %s)!\n", cudaGetErrorString(err));
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
