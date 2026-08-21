/** * ====================================================================
 * @file        lib_opencl_common.cpp (./cifar_10_multiple)
 * @brief       Common OpenCL platform initialization, device setup, 
 *              profiling timer evaluation, and generic cleanup routines.
 * @paragraph   License
 * ESA-PL Strong Copyleft – v2.5
 * ======================================================================= */
#include "../benchmark_library.h"
#include "../../common/opencl_common.hpp"
#include <cstring>

void init(GraficCommon* device_object, char* device_name){
	init(device_object, 0,0, device_name);
}
void init(GraficCommon* device_object, int platform ,int device, char* device_name){
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	//get all platforms (drivers)
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if(all_platforms.size()==0){
        std::cout<<" No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Platform default_platform=all_platforms[platform];
    //std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";
   //get default device of the default platform
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if(all_devices.size()==0){
        std::cout<<" No devices found. Check OpenCL installation!\n";
        exit(1);
    }

    cl::Device default_device=all_devices[device];
    //std::cout<< "Using device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<"\n";
    strcpy(device_name,default_device.getInfo<CL_DEVICE_NAME>().c_str() );
    // context
    deviceObj->context = new cl::Context(default_device);
    deviceObj->queue = new cl::CommandQueue(*deviceObj->context,default_device,CL_QUEUE_PROFILING_ENABLE);
    deviceObj->default_device = default_device;
    
    // events
    deviceObj->evt_copyIN = new cl::Event;
    deviceObj->evt_copyK1 = new cl::Event;
    deviceObj->evt_copyK2 = new cl::Event;
    deviceObj->evt_copyW1 = new cl::Event;
    deviceObj->evt_copyW2 = new cl::Event;
    deviceObj->evt_copyOut = new cl::Event;

    deviceObj->evt1_1 = new cl::Event;
    deviceObj->evt1_2 = new cl::Event; 
    deviceObj->evt1_3 = new cl::Event; 
    deviceObj->evt1_4 = new cl::Event; 
    deviceObj->evt2_1 = new cl::Event;  
    deviceObj->evt2_2 = new cl::Event; 
    deviceObj->evt2_3 = new cl::Event;
    deviceObj->evt2_1 = new cl::Event;  
    deviceObj->evt2_2 = new cl::Event; 
    deviceObj->evt2_3 = new cl::Event; 
    deviceObj->evt2_4 = new cl::Event;  
    deviceObj->evtd_1 = new cl::Event; 
    deviceObj->evtd_1_a = new cl::Event;
    deviceObj->evtd_2 = new cl::Event; 
    deviceObj->evtd_2_a = new cl::Event;  
    deviceObj->evt_softmax = new cl::Event;
    deviceObj->evt_softmax_fin = new cl::Event;   
}

bool device_memory_init(GraficCommon* device_object, unsigned int input_data, unsigned int output_data, unsigned int kernel_1, unsigned int kernel_2, unsigned int stride_1, unsigned int stride_2, unsigned int neurons_dense_1, unsigned int neurons_dense_2, unsigned int number_of_images){
   GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
   cl_int err;

   unsigned int size_pooling_1 = input_data / stride_1;
   unsigned int size_pooling_2 = size_pooling_1 / stride_2;
   unsigned int weights_layer_1 = size_pooling_2 * size_pooling_2 * neurons_dense_1;
   unsigned int weights_layer_2 = neurons_dense_1 * neurons_dense_2; 

   // input
   deviceObj->input_data = new cl::Buffer(*deviceObj->context,CL_MEM_READ_ONLY ,number_of_images * input_data * input_data * sizeof(bench_t), nullptr, &err);
   if (err != CL_SUCCESS) return false;
   
   // kernel 1
   deviceObj->kernel_1 = new cl::Buffer(*deviceObj->context,CL_MEM_READ_ONLY ,kernel_1 * kernel_1 * sizeof(bench_t), nullptr, &err);
   if (err != CL_SUCCESS) return false;

   // convolution 1
   deviceObj->conv_1_output = new cl::Buffer(*deviceObj->context,CL_MEM_READ_WRITE , NUMBER_OF_STREAMS * input_data * input_data * sizeof(bench_t), nullptr, &err);
   if (err != CL_SUCCESS) return false;
   
   // pooling 1 
   deviceObj->pooling_1_output = new cl::Buffer(*deviceObj->context,CL_MEM_READ_WRITE , NUMBER_OF_STREAMS * size_pooling_1 * size_pooling_1 * sizeof(bench_t), nullptr, &err);
   if (err != CL_SUCCESS) return false;
  
   // convolution 2
   deviceObj->kernel_2 = new cl::Buffer(*deviceObj->context,CL_MEM_READ_ONLY ,kernel_2 * kernel_2 * sizeof(bench_t), nullptr, &err);
   if (err != CL_SUCCESS) return false;

   // conv 2 output
   deviceObj->conv_2_output = new cl::Buffer(*deviceObj->context,CL_MEM_READ_WRITE , NUMBER_OF_STREAMS * size_pooling_1 * size_pooling_1 * sizeof(bench_t), nullptr, &err);
   if (err != CL_SUCCESS) return false;
   
   // pooling 2 
   deviceObj->pooling_2_output = new cl::Buffer(*deviceObj->context,CL_MEM_READ_WRITE , NUMBER_OF_STREAMS * size_pooling_2 * size_pooling_2 * sizeof(bench_t), nullptr, &err);
   if (err != CL_SUCCESS) return false;
   
   // dense 1 weights
   deviceObj->dense_layer_1_weights = new cl::Buffer(*deviceObj->context,CL_MEM_READ_ONLY ,weights_layer_1 * sizeof(bench_t), nullptr, &err);
   if (err != CL_SUCCESS) return false;

   // dense 1 output 
   deviceObj->dense_layer_1_output = new cl::Buffer(*deviceObj->context,CL_MEM_READ_WRITE , NUMBER_OF_STREAMS * neurons_dense_1 * sizeof(bench_t), nullptr, &err);
   if (err != CL_SUCCESS) return false;
   
   // dense 2 weights
   deviceObj->dense_layer_2_weights = new cl::Buffer(*deviceObj->context,CL_MEM_READ_ONLY ,weights_layer_2 * sizeof(bench_t), nullptr, &err);
   if (err != CL_SUCCESS) return false;

   // dense 2 output 
   deviceObj->dense_layer_2_output = new cl::Buffer(*deviceObj->context,CL_MEM_READ_WRITE , NUMBER_OF_STREAMS * neurons_dense_2 * sizeof(bench_t), nullptr, &err);
   if (err != CL_SUCCESS) return false;
   
   // out
   deviceObj->sum_ouput = new cl::Buffer(*deviceObj->context,CL_MEM_READ_WRITE , NUMBER_OF_STREAMS * sizeof(bench_t), nullptr, &err);
   if (err != CL_SUCCESS) return false;

   deviceObj->output_data = new cl::Buffer(*deviceObj->context,CL_MEM_READ_WRITE ,number_of_images * neurons_dense_2 * sizeof(bench_t), nullptr, &err);
   if (err != CL_SUCCESS) return false;
   
   return true;
}

void copy_memory_to_device(GraficCommon* device_object, bench_t* input_data, bench_t* kernel_1_data, bench_t* kernel_2_data, bench_t* weights_1 ,bench_t* weights_2,unsigned int input , unsigned int kernel_size_1, unsigned int kernel_size_2, unsigned int weights_1_size, unsigned int weights_2_size, unsigned int number_of_images){
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	// host -> device
    Clock h2dCLK;

    // Clock profilling start 
    h2dCLK.start();

    // input data
    cl_int err = deviceObj->queue->enqueueWriteBuffer(*deviceObj->input_data,CL_TRUE,0,sizeof(bench_t)* input * input * number_of_images, input_data, NULL, deviceObj->evt_copyIN);
    if (openclError("Failed to copy input_data from host to device", err)) return;
    
    // kernels
    err = deviceObj->queue->enqueueWriteBuffer(*deviceObj->kernel_1,CL_TRUE,0,sizeof(bench_t)* kernel_size_1 * kernel_size_1, kernel_1_data, NULL, deviceObj->evt_copyK1);
    if (openclError("Failed to copy kernel_1 from host to device", err)) return;

    err = deviceObj->queue->enqueueWriteBuffer(*deviceObj->kernel_2,CL_TRUE,0,sizeof(bench_t)* kernel_size_2 * kernel_size_2, kernel_2_data, NULL, deviceObj->evt_copyK2);
    if (openclError("Failed to copy kernel_2 from host to device", err)) return;

    // dense layer
    err = deviceObj->queue->enqueueWriteBuffer(*deviceObj->dense_layer_1_weights,CL_TRUE,0,sizeof(bench_t)* weights_1_size, weights_1, NULL, deviceObj->evt_copyW1);
    if (openclError("Failed to copy dense_layer_1_weights from host to device", err)) return;

    err = deviceObj->queue->enqueueWriteBuffer(*deviceObj->dense_layer_2_weights,CL_TRUE,0,sizeof(bench_t)* weights_2_size, weights_2, NULL, deviceObj->evt_copyW2);
    if (openclError("Failed to copy dense_layer_2 from host to device", err)) return;

    // Clock profilling end 
    h2dCLK.end();

    // store the hd2h time
    deviceObj->h2d_elapsed_time = h2dCLK.getElapsedNS();
}


void copy_memory_to_host(GraficCommon* device_object, bench_t* h_C, int size, unsigned int number_of_images){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    // device ->  host
    Clock d2hCLK;

    // Clock profilling start 
    d2hCLK.start();

    cl_int err = deviceObj->queue->enqueueReadBuffer(*deviceObj->output_data, CL_TRUE, 0, sizeof(bench_t)*size*number_of_images, h_C, NULL, deviceObj->evt_copyOut);
    if (openclError("Failed to copy vector output_data from device to host", err)) return;
    //deviceObj->queue->enqueueReadBuffer(*deviceObj->conv_2_output,CL_TRUE,0,sizeof(bench_t)*16*16,h_C, NULL, deviceObj->evt_copyOut);
    
    // Clock profilling end 
    d2hCLK.end();
    
    // store the hd2h time
    deviceObj->d2h_elapsed_time = d2hCLK.getElapsedNS();
}

float get_elapsed_time(GraficCommon* device_object, bool csv_format,bool csv_format_timestamp, long int current_time)
{
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    deviceObj->evt_copyOut->wait();

    float elapsed_h_d = 0, elapsed = 0, elapsed_d_h = 0;
    
    if (deviceObj->profiling_clock)
    {
        // --- FIX: Use <chrono> instead of CLBlast event profiling (unreliable on PROFILING_CLOCK) ---
        elapsed_h_d  = deviceObj->h2d_elapsed_time;
        elapsed      = deviceObj->elapsed_time;
        elapsed_d_h  = deviceObj->d2h_elapsed_time;
    }else{
        
        // copy memory H -> D
        elapsed_h_d = deviceObj->evt_copyIN->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt_copyIN->getProfilingInfo<CL_PROFILING_COMMAND_START>();
        elapsed_h_d += deviceObj->evt_copyK1->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt_copyK1->getProfilingInfo<CL_PROFILING_COMMAND_START>();
        elapsed_h_d += deviceObj->evt_copyK2->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt_copyK2->getProfilingInfo<CL_PROFILING_COMMAND_START>();
        elapsed_h_d += deviceObj->evt_copyW1->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt_copyW1->getProfilingInfo<CL_PROFILING_COMMAND_START>();
        elapsed_h_d += deviceObj->evt_copyW2->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt_copyW2->getProfilingInfo<CL_PROFILING_COMMAND_START>();
        
        // kernel time
        elapsed = deviceObj->evt_softmax_fin->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt1_1->getProfilingInfo<CL_PROFILING_COMMAND_START>();
        
        // copy memory D -> H
        elapsed_d_h = deviceObj->evt_copyOut->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt_copyOut->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    }

    if (csv_format_timestamp){
        printf("%.10f;%.10f;%.10f;%ld;\n", elapsed_h_d / 1000000.0,deviceObj->elapsed_time,elapsed_d_h / 1000000.0, current_time);
    }
    else if (csv_format){
         printf("%.10f;%.10f;%.10f;\n", elapsed_h_d / 1000000.0,deviceObj->elapsed_time,elapsed_d_h / 1000000.0);
    }else{
         printf("profiling mode: %s\n", deviceObj->profiling_clock ? "CLOCK" : "GPU");
         printf("Elapsed time Host->Device: %.10f milliseconds\n", (elapsed_h_d / 1000000.0));
         printf("Elapsed time kernel: %.10f milliseconds\n", elapsed / 1000000.0);
         printf("Elapsed time Device->Host: %.10f milliseconds\n", elapsed_d_h / 1000000.0);
    }
    return elapsed / 1000000.0; // TODO Change
}

void clean(GraficCommon* device_object){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);

    // pointers clean
    delete deviceObj->context;
    delete deviceObj->queue;
    // pointer to memory

    delete deviceObj->evt_copyIN;
    delete deviceObj->evt_copyK1;
    delete deviceObj->evt_copyK2;
    delete deviceObj->evt_copyW1;
    delete deviceObj->evt_copyW2;
    delete deviceObj->evt_copyOut;
    delete deviceObj->evt1_1;
    delete deviceObj->evt1_2;
    delete deviceObj->evt1_3;
    delete deviceObj->evt1_4;
    delete deviceObj->evt2_1;
    delete deviceObj->evt2_2;
    delete deviceObj->evt2_3;
    delete deviceObj->evt2_4;
    delete deviceObj->evtd_1;
    delete deviceObj->evtd_1_a;
    delete deviceObj->evtd_2;
    delete deviceObj->evtd_2_a;
    delete deviceObj->evt_softmax;
    delete deviceObj->evt_softmax_fin;

    delete deviceObj->input_data;
    delete deviceObj->kernel_1;
    delete deviceObj->conv_1_output;
    delete deviceObj->pooling_1_output;
    delete deviceObj->kernel_2;
    delete deviceObj->conv_2_output;
    delete deviceObj->pooling_2_output;
    delete deviceObj->dense_layer_1_weights;
    delete deviceObj->dense_layer_1_output;
    delete deviceObj->dense_layer_2_weights;
    delete deviceObj->dense_layer_2_output;
    delete deviceObj->output_data;
    delete deviceObj->sum_ouput;
}

#ifdef UMA_COMPATIBILITY
// ====== UMA function ======
void get_unified_memory_pointers(GraficCommon* device_object,bench_t* &input_data, unsigned int input_mem_size,bench_t* &kernel_1, bench_t* &kernel_2, unsigned int kernel_mem_size, bench_t* &weights_1, unsigned int weights_1_mem_size, bench_t* &weights_2, unsigned int weights_2_mem_size, bench_t* &d_output, unsigned int output_mem_size){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    // --- Call the openCL common function ---
     map_unified_memory(device_object, input_mem_size,
        BufferMapCL{&input_data, deviceObj->input_data, nullptr});

    map_unified_memory(device_object, kernel_mem_size,
        BufferMapCL{&kernel_1, deviceObj->kernel_1, nullptr},
        BufferMapCL{&kernel_2, deviceObj->kernel_2, nullptr});

    map_unified_memory(device_object, weights_1_mem_size,
        BufferMapCL{&weights_1, deviceObj->dense_layer_1_weights, nullptr});

    map_unified_memory(device_object, weights_2_mem_size,
        BufferMapCL{&weights_2, deviceObj->dense_layer_2_weights, nullptr});

    map_unified_memory(device_object, output_mem_size,
        BufferMapCL{&d_output, deviceObj->output_data, nullptr});
}

void sync_unified_memory_to_device(GraficCommon* device_object, bench_t* &input_data, bench_t* &kernel_1, bench_t* &kernel_2, bench_t* &weights_1, bench_t* &weights_2, bench_t* &d_output){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    // --- Call the openCL common function ---
    unmap_unified_memory(device_object,
        BufferMapCL{&input_data, deviceObj->input_data,             deviceObj->evt_copyIN},
        BufferMapCL{&kernel_1,   deviceObj->kernel_1,               deviceObj->evt_copyK1},
        BufferMapCL{&kernel_2,   deviceObj->kernel_2,               deviceObj->evt_copyK2},
        BufferMapCL{&weights_1,  deviceObj->dense_layer_1_weights,  deviceObj->evt_copyW1},
        BufferMapCL{&weights_2,  deviceObj->dense_layer_2_weights,  deviceObj->evt_copyW2},
        BufferMapCL{&d_output,   deviceObj->output_data,            nullptr}
    );
} 

void sync_unified_memory_to_host(GraficCommon* device_object, bench_t* &d_output, unsigned int size_output){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    // --- Call the openCL common function ---
    map_unified_memory_to_host(device_object, size_output, 
        BufferMapCL{&d_output, deviceObj->output_data, deviceObj->evt_copyOut}
    );
}

#endif