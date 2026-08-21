// OpenCL lib code 
#include <cmath>
#include "../benchmark_library.h"
#include <cstring>
#include "kernel.cl"
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
    deviceObj->evt_copyC = new cl::Event;
    
}

bool device_memory_init(GraficCommon* device_object, unsigned int size_a_matrix, unsigned int size_b_matrix, unsigned int size_c_matrix){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    cl_int err;

    deviceObj->d_A = new cl::Buffer(*deviceObj->context,CL_MEM_READ_ONLY ,sizeof(bench_t)*size_a_matrix, nullptr, &err);
    if (err != CL_SUCCESS) return false;

    deviceObj->d_B = new cl::Buffer(*deviceObj->context,CL_MEM_READ_ONLY ,sizeof(bench_t)*size_b_matrix, nullptr, &err);
    if (err != CL_SUCCESS) return false;

    deviceObj->kernel = new cl::Buffer(*deviceObj->context,CL_MEM_READ_WRITE ,sizeof(bench_t)*size_c_matrix, nullptr, &err);
    if (err != CL_SUCCESS) return false;
    
    // inicialice Arrays
    return true;
}

void copy_memory_to_device(GraficCommon* device_object, bench_t* h_A, bench_t* kernel, unsigned int size_a, unsigned int size_b){
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	// host -> device
    Clock h2dCLK;

    // Clock profilling start 
    h2dCLK.start();

    cl_int err = deviceObj->queue->enqueueWriteBuffer(*deviceObj->d_A,CL_TRUE,0,sizeof(bench_t)*size_a, h_A, NULL, deviceObj->evt_copyA);
    if (openclError("Failed to copy data vector B from host to device", err)) return;
    
    err = deviceObj->queue->enqueueWriteBuffer(*deviceObj->kernel,CL_TRUE,0,sizeof(bench_t)*size_b, kernel, NULL, deviceObj->evt_copyB);
    if (openclError("Failed to copy kernel data from host to device", err)) return;

    // Clock profilling end 
    h2dCLK.end();

    // store the hd2h time
    deviceObj->h2d_elapsed_time = h2dCLK.getElapsedNS();
}


void execute_kernel(GraficCommon* device_object, unsigned int n, unsigned int m,unsigned int w, unsigned int kernel_size){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    //FIX: use W instead of  N the other implementation need size_matrix and this one need size_B
    n = w;
    const unsigned int x_local= BLOCK_SIZE;
    cl::NDRange local;
    cl::NDRange global;
    if (n < BLOCK_SIZE)
    {
        local = cl::NullRange;
        global = cl::NDRange(m + x_local);
    }
    else
    {
        local = cl::NDRange(x_local);
        global = cl::NDRange(m + x_local);
    }
    

    cl::Program::Sources sources;
    deviceObj->evt = new cl::Event;
    // load kernel from file
    kernel_code = type_kernel_common + kernel_code;
    sources.push_back({kernel_code.c_str(),kernel_code.length()});

    cl::Program program(*deviceObj->context,sources);
    if(program.build({deviceObj->default_device})!=CL_SUCCESS){
        std::cout<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(deviceObj->default_device)<<"\n";
        exit(1);
    }

    // kernel time execution
    Clock kernelCLK;

    // Clock profilling start 
    kernelCLK.start();

    cl::Kernel kernel_conv=cl::Kernel(program,"kernel_vector_convolution");
    kernel_conv.setArg(0,*deviceObj->d_A);
    kernel_conv.setArg(1,*deviceObj->d_B);
    kernel_conv.setArg(2,*deviceObj->kernel);
    kernel_conv.setArg(3,n);
    kernel_conv.setArg(4,m);
    kernel_conv.setArg(5,w);
    kernel_conv.setArg(6,kernel_size);

    
    deviceObj->queue->enqueueNDRangeKernel(kernel_conv,cl::NullRange,global,local, NULL, deviceObj->evt);
    
    // Wait for completion before stopping the clock
    deviceObj->queue->finish();
    // Clock profilling end 
    kernelCLK.end();

    // store the kernel time
    deviceObj->elapsed_time = kernelCLK.getElapsedNS();
}

void copy_memory_to_host(GraficCommon* device_object, bench_t* h_C, int size){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    // device ->  host
    Clock d2hCLK;

    // Clock profilling start 
    d2hCLK.start();

    cl_int err = deviceObj->queue->enqueueReadBuffer(*deviceObj->d_B, CL_TRUE, 0, sizeof(bench_t)*size, h_C, NULL, deviceObj->evt_copyC);
    if (openclError("Failed to copy vector B from device to host", err)) return;

    // Clock profilling end 
    d2hCLK.end();
    
    // store the hd2h time
    deviceObj->d2h_elapsed_time = d2hCLK.getElapsedNS();
}

float get_elapsed_time(GraficCommon* device_object, bool csv_format,bool csv_format_timestamp, long int current_time){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    deviceObj->evt_copyC->wait();
    
    float elapsed_h_d = 0, elapsed = 0, elapsed_d_h = 0;
    
    if (deviceObj->profiling_clock)
    {
        // --- FIX: Use <chrono> instead of CLBlast event profiling (unreliable on PROFILING_CLOCK) ---
        elapsed_h_d  = deviceObj->h2d_elapsed_time;
        elapsed      = deviceObj->elapsed_time;
        elapsed_d_h  = deviceObj->d2h_elapsed_time;
    }else{
        elapsed_h_d = deviceObj->evt_copyA->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt_copyA->getProfilingInfo<CL_PROFILING_COMMAND_START>();
        elapsed_h_d += deviceObj->evt_copyB->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt_copyB->getProfilingInfo<CL_PROFILING_COMMAND_START>();
        //printf("Elapsed time Host->Device: %.10f \n", elapsed / 1000000.0);
        elapsed = deviceObj->evt->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt->getProfilingInfo<CL_PROFILING_COMMAND_START>();
        //printf("Elapsed time kernel: %.10f \n", elapsed / 1000000.0);
        elapsed_d_h = deviceObj->evt_copyC->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt_copyC->getProfilingInfo<CL_PROFILING_COMMAND_START>();
        //printf("Elapsed time Device->Host: %.10f \n", );
    }


    if (csv_format_timestamp){
         printf("%.10f;%.10f;%.10f;%ld;\n", elapsed_h_d / 1000000.0,elapsed / 1000000.0,elapsed_d_h / 1000000.0, current_time);
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
    delete deviceObj->kernel;
    delete deviceObj->evt;
    delete deviceObj->evt_copyA;
    delete deviceObj->evt_copyB;
    delete deviceObj->evt_copyC;
}


#ifdef UMA_COMPATIBILITY
// ====== UMA function ======
void get_unified_memory_pointers(GraficCommon* device_object, bench_t* &A, bench_t* &B, bench_t* &C, unsigned sizeA, unsigned sizeB, unsigned sizeC){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    // --- Call the openCL common function ---
    // input
    map_unified_memory(device_object, sizeA, 
        BufferMapCL{&A, deviceObj->d_A, nullptr}
    );

    // kernel
    map_unified_memory(device_object, sizeB, 
        BufferMapCL{&B, deviceObj->kernel, nullptr}
    );

    // output 
    map_unified_memory(device_object, sizeC, 
        BufferMapCL{&C, deviceObj->d_B, nullptr}
    );
}

void sync_unified_memory_to_device(GraficCommon* device_object, bench_t* &A, bench_t* &B, bench_t* &C){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    // --- Call the openCL common function ---
    unmap_unified_memory(device_object, 
        BufferMapCL{&A, deviceObj->d_A, deviceObj->evt_copyA},
        BufferMapCL{&B, deviceObj->kernel, deviceObj->evt_copyB},
        BufferMapCL{&C, deviceObj->d_B, nullptr}
    );
} 


void sync_unified_memory_to_host(GraficCommon* device_object, bench_t* &d_output, unsigned int size_output){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    // --- Call the openCL common function ---
    map_unified_memory_to_host(device_object, size_output, 
        BufferMapCL{&d_output, deviceObj->d_B, deviceObj->evt_copyC}
    );
}

#endif