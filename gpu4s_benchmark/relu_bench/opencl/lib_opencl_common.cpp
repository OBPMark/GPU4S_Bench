/** * ====================================================================
 * @file        lib_opencl_common.cpp (./relu_bench)
 * @brief       Common OpenCL platform initialization, device setup, 
 *              profiling timer evaluation, and generic cleanup routines.
 * @paragraph   License
 * ESA-PL Strong Copyleft – v2.5
 * ======================================================================= */
#include "../benchmark_library.h"
#include "../../common/opencl_common.hpp"


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


bool device_memory_init(GraficCommon* device_object, unsigned int size_a_matrix, unsigned int size_b_matrix){
   GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
   cl_int err;
   deviceObj->d_A = new cl::Buffer(*deviceObj->context,CL_MEM_READ_ONLY ,sizeof(bench_t)*size_a_matrix, nullptr, &err);
   if (err != CL_SUCCESS) return false;

   deviceObj->d_B = new cl::Buffer(*deviceObj->context,CL_MEM_READ_WRITE ,sizeof(bench_t)*size_b_matrix, nullptr, &err);
   if (err != CL_SUCCESS) return false;
   
   // inicialice Arrays
   return true;
}

void copy_memory_to_device(GraficCommon* device_object, bench_t* h_A, unsigned int size_a){
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	// host -> device
    Clock h2dCLK;

    // Clock profilling start 
    h2dCLK.start();
    
    // Enqueue writing host memory h_A to device buffer d_A
    cl_int err = deviceObj->queue->enqueueWriteBuffer(*deviceObj->d_A,CL_TRUE,0,sizeof(bench_t)*size_a, h_A, NULL, deviceObj->evt_copyA);
    if (openclError("Failed to copy vector A from host to device", err)) return;

    // Clock profilling end 
    h2dCLK.end();

    // store the hd2h time
    deviceObj->h2d_elapsed_time = h2dCLK.getElapsedNS();
}


void copy_memory_to_host(GraficCommon* device_object, bench_t* h_C, int size){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    // device ->  host
    Clock d2hCLK;

    // Clock profilling start 
    d2hCLK.start();

    cl_int err = deviceObj->queue->enqueueReadBuffer(*deviceObj->d_B, CL_TRUE, 0, sizeof(bench_t)*size, h_C, NULL, deviceObj->evt_copyB);
    if (openclError("Failed to copy vector B from device to host", err)) return;

    // Clock profilling end 
    d2hCLK.end();
    
    // store the hd2h time
    deviceObj->d2h_elapsed_time = d2hCLK.getElapsedNS();
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
        
        elapsed = deviceObj->evt->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt->getProfilingInfo<CL_PROFILING_COMMAND_START>();
        //printf("Elapsed time kernel: %.10f \n", elapsed / 1000000.0);
        
        elapsed_d_h = deviceObj->evt_copyB->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt_copyB->getProfilingInfo<CL_PROFILING_COMMAND_START>();
        //printf("Elapsed time Device->Host: %.10f \n", );
    }

    #ifdef UNIFIED_MEMORY
        elapsed_h_d = h2dTotal;
    #endif


    if (csv_format_timestamp){
        printf("%.10f;%.10f;%.10f;%ld;\n",  elapsed_h_d / 1000000.0,elapsed / 1000000.0,elapsed_d_h / 1000000.0,current_time);
    }
    else if (csv_format){
         printf("%.10f;%.10f;%.10f;\n", elapsed_h_d / 1000000.0,elapsed / 1000000.0,elapsed_d_h / 1000000.0);
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
    delete deviceObj->d_A;
    delete deviceObj->d_B;
    delete deviceObj->evt;
    delete deviceObj->evt_copyA;
    delete deviceObj->evt_copyB;
}

#ifdef UMA_COMPATIBILITY
// ====== UMA function ======
void get_unified_memory_pointers(GraficCommon* device_object, bench_t* &A, bench_t* &B, unsigned int memSize){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    // --- Call the openCL common function ---
    map_unified_memory(device_object, memSize, 
        BufferMapCL{&A, deviceObj->d_A, nullptr},
        BufferMapCL{&B, deviceObj->d_B, nullptr}
    );
}

void sync_unified_memory_to_device(GraficCommon* device_object, bench_t* &A, bench_t* &B){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    // --- Call the openCL common function ---
    unmap_unified_memory(device_object, 
        BufferMapCL{&A, deviceObj->d_A, deviceObj->evt_copyA},
        BufferMapCL{&B, deviceObj->d_B, nullptr}
    );
} 


void sync_unified_memory_to_host(GraficCommon* device_object, bench_t* &d_output, unsigned int size_output){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    // --- Call the openCL common function ---
    map_unified_memory_to_host(device_object, size_output, 
        BufferMapCL{&d_output, deviceObj->d_B, deviceObj->evt_copyB}
    );
}

#endif
