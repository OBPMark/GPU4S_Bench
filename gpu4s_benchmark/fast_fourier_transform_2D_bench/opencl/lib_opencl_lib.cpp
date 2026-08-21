// OpenCL lib code 
#include <cmath>
#include "../benchmark_library.h"
#include "../../common/opencl_common.hpp"
#include "vkFFT.h"

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
    deviceObj->evt = new cl::Event; 
    deviceObj->evt_copyA = new cl::Event;
    deviceObj->evt_copyB = new cl::Event;
}

bool device_memory_init(GraficCommon* device_object, int64_t size){
   GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
   cl_int err;

   deviceObj->d_A = new cl::Buffer(*deviceObj->context,CL_MEM_READ_ONLY ,sizeof(bench_t)* size * size * 2, nullptr, &err);
   if (err != CL_SUCCESS) return false;
   
   deviceObj->d_B = new cl::Buffer(*deviceObj->context,CL_MEM_READ_WRITE ,sizeof(bench_t)* size * size * 2, nullptr, &err);
   if (err != CL_SUCCESS) return false;
   
   // inicialice Arrays
   return true;
}

void copy_memory_to_device(GraficCommon *device_object, COMPLEX **h_A,int64_t size){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    // --- init ---
    bench_t *h_signal = (bench_t *)malloc(sizeof(bench_t) * size * size * 2);
    for (int i=0; i<size; ++i)
        {
            for (int j=0; j<size; ++j)
            {
                    h_signal[2*(j+i*size)] = h_A[i][j].x ;
                    h_signal[2*(j+i*size)+1] = h_A[i][j].y;
            }
        }

    // host -> device
    Clock h2dCLK;

    // Clock profilling start 
    h2dCLK.start();

    // copy memory host -> device
    deviceObj->queue->enqueueWriteBuffer(*deviceObj->d_A,CL_TRUE,0,sizeof(bench_t)*size*size*2, h_signal, NULL, deviceObj->evt_copyA);

    // Clock profilling end 
    h2dCLK.end();

    // store the hd2h time
    deviceObj->h2d_elapsed_time = h2dCLK.getElapsedNS();
    free(h_signal);
}

void execute_kernel(GraficCommon* device_object, int64_t size) {
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);

    // --- convert c++ pointer to raw c ---
    cl_context      raw_context = (*deviceObj->context)();
    cl_device_id    raw_device  = deviceObj->default_device();
    cl_command_queue raw_queue  = (*deviceObj->queue)();
    cl_mem raw_input            = (*deviceObj->d_A)();
    cl_mem raw_output           = (*deviceObj->d_B)();

    // --- VkFFT configuration ---
    VkFFTConfiguration config = {};
    config.FFTdim           = 2;            // number of dim of FFT
    config.size[0]          = size;         // size of D1
    config.size[1]          = size;         // size of D2
    config.device           = &raw_device;  // select the device 
    config.context          = &raw_context; // memory space
    config.buffer           = &raw_output;  // output buff
    config.inputBuffer      = &raw_input;   // input buff  
    config.isInputFormatted = 1;            // different buffer for input/output
    #ifdef DOUBLE
    config.doublePrecision  = 1;
    #endif
    
    #ifdef ANDROID
    // --- ADRENO MOBILE HARDWARE CONSTRAINTS ---
    // FIX: Reduce the chunk size for shared (local) memory reads (Adreno has very little local RAM)
    config.coalescedMemory = 32; 

    // FIX: Lower the target threads per block and precompute math (LUT)
    config.aimThreads = 64;  
    #endif

    // kernel time execution
    Clock kernelCLK;

    // --- init ---
    kernelCLK.start(); // Start clock
    VkFFTApplication app = {};
    initializeVkFFT(&app, config); // compile the FFT kernel for your GPU

    // --- launch ---
    VkFFTLaunchParams launchParams = {};
    launchParams.commandQueue  = &raw_queue; //select the queue

    VkFFTAppend(&app, -1, &launchParams); // -1 = forward FFT

    deviceObj->queue->finish();
    kernelCLK.end(); // End clock

    // store the kernel time
    deviceObj->elapsed_time = kernelCLK.getElapsedNS();

    // --- cleanup ---
    deleteVkFFT(&app);
}

void copy_memory_to_host(GraficCommon* device_object, COMPLEX **h_B, int64_t size){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    bench_t *h_signal = (bench_t *)malloc(sizeof(bench_t) * size * size * 2);

    // device ->  host
    Clock d2hCLK;

    // Clock profilling start 
    d2hCLK.start();
    
    cl_int err = deviceObj->queue->enqueueReadBuffer(*deviceObj->d_B, CL_TRUE, 0, sizeof(bench_t)*size*size * 2, h_signal, NULL, deviceObj->evt_copyB);
    if (openclError("Failed to copy vector B from device to host", err)) return;
    
    // Clock profilling end 
    d2hCLK.end();
    
    // store the hd2h time
    deviceObj->d2h_elapsed_time = d2hCLK.getElapsedNS();
    
    for (int i=0; i<size; ++i){
        for (int j=0; j<size; ++j){
                h_B[i][j].x = h_signal[2*(j+i*size)];
                h_B[i][j].y = h_signal[2*(j+i*size)+1];
        }
    }
    free(h_signal);
}

float get_elapsed_time(GraficCommon* device_object, bool csv_format, bool csv_format_timestamp, long int current_time){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    deviceObj->evt_copyB->wait();

    float elapsed_h_d = 0, elapsed = 0, elapsed_d_h = 0;
    
    if (deviceObj->profiling_clock)
    {
        // --- FIX: Use <chrono> instead of CLBlast event profiling (unreliable on PROFILING_CLOCK) ---
        elapsed_h_d  = deviceObj->h2d_elapsed_time;
        elapsed      = deviceObj->elapsed_time;
        elapsed_d_h  = deviceObj->d2h_elapsed_time;
    }else{
        elapsed_h_d = deviceObj->evt_copyA->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt_copyA->getProfilingInfo<CL_PROFILING_COMMAND_START>();
        //printf("Elapsed time Host->Device: %.10f \n", elapsed / 1000000.0);
        elapsed_d_h = deviceObj->evt_copyB->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt_copyB->getProfilingInfo<CL_PROFILING_COMMAND_START>();
        //printf("Elapsed time Device->Host: %.10f \n", );
        elapsed      = deviceObj->elapsed_time;
    }

    if (csv_format_timestamp){
        printf("%.10f;%.10f;%.10f;%ld;\n", elapsed_h_d / 1000000.0, elapsed / 1000000.0,elapsed_d_h / 1000000.0, current_time);
    }
    else if (csv_format){
         printf("%.10f;%.10f;%.10f;\n", elapsed_h_d / 1000000.0, elapsed / 1000000.0,elapsed_d_h / 1000000.0);
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
    // pointer to memory
    delete deviceObj->d_A;
    delete deviceObj->d_B;
    delete deviceObj->evt;
    delete deviceObj->evt_copyA;
    delete deviceObj->evt_copyB;
}


#ifdef UMA_COMPATIBILITY
// ====== UMA function ======
void get_unified_memory_pointers(GraficCommon* device_object, COMPLEX** &A, COMPLEX** &B, int64_t memSize){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    int64_t flatSize = sizeof(bench_t) * memSize * memSize * 2;

    //Map temp 2D buffer
    bench_t* tmp_flat_A = nullptr;
    bench_t* tmp_flat_B = nullptr;

    // --- Call the openCL common function ---
    map_unified_memory(device_object, flatSize, 
        BufferMapCL{&tmp_flat_A, deviceObj->d_A, nullptr},
        BufferMapCL{&tmp_flat_B, deviceObj->d_B, nullptr}
    );

    // --- Cast back to bench_t ---
    COMPLEX* flat_A = (COMPLEX*)tmp_flat_A;
    COMPLEX* flat_B = (COMPLEX*)tmp_flat_B;

    // --- 2D -> 1D ---
    //Connect ptr 2D array to the flat ptr
    for (int i = 0; i < memSize; ++i){
        A[i] = flat_A + (i * memSize);
        B[i] = flat_B + (i * memSize);
    }
}

void sync_unified_memory_to_device(GraficCommon* device_object, COMPLEX** &A, COMPLEX** &B){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);

    bench_t* tmp_flat_A = (bench_t*)A[0];
    bench_t* tmp_flat_B = (bench_t*)B[0];

    // --- Call the openCL common function ---
    unmap_unified_memory(device_object, 
        BufferMapCL{&tmp_flat_A, deviceObj->d_A, deviceObj->evt_copyA},
        BufferMapCL{&tmp_flat_B, deviceObj->d_B, nullptr}
    );
} 


void sync_unified_memory_to_host(GraficCommon* device_object, COMPLEX** &d_output, int64_t memSize){    
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    int64_t flatSize = sizeof(bench_t) * memSize * memSize * 2;
    bench_t* tmp_flat_B = nullptr;

    // --- Call the openCL common function ---
    map_unified_memory_to_host(device_object, flatSize, 
        BufferMapCL{&tmp_flat_B, deviceObj->d_B, deviceObj->evt_copyB}
    );

    // --- Cast back to 2D complex buffer ---
    COMPLEX* flat_B = (COMPLEX*)tmp_flat_B;

    // Reconnect to 2D pointer array to the newly mapped output memory
    for (int i = 0; i < memSize; ++i){
        d_output[i] = flat_B + (i * memSize);
    }
}
#endif