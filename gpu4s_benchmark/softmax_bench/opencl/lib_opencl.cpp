// OpenCL lib code 
#include <cmath>
#include "../benchmark_library.h"
#include <cstring>
#include "GEN_kernel.hcl"
#include "GEN_atomic_functions.hcl"

void execute_kernel(GraficCommon* device_object, unsigned int n, unsigned int m, unsigned int w){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    const unsigned int x_local= BLOCK_SIZE;
    const unsigned int y_local= BLOCK_SIZE;
    cl::NDRange local, global;
    if(n < BLOCK_SIZE)
    {
        local = cl::NDRange (1, 1);
        global = cl::NDRange (n, w);
    }
    else
    {
        local = cl::NDRange(x_local, y_local);
        global = cl::NDRange(n, w);
    }

    cl::Program::Sources sources;
    // load kernel from file
    kernel_code = type_kernel_common + atomic_code + kernel_code;
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


    cl::Kernel softmax_kernel=cl::Kernel(program,"kernel_softmax");
    softmax_kernel.setArg(0,*deviceObj->d_A);
    softmax_kernel.setArg(1,*deviceObj->d_B);
    softmax_kernel.setArg(2,*deviceObj->sum_d_B);
    softmax_kernel.setArg(3,n);

    //deviceObj->queue->finish();
    cl::Kernel softmax_end_kernel=cl::Kernel(program,"kernel_softmax_end");
    softmax_end_kernel.setArg(0,*deviceObj->d_B);
    softmax_end_kernel.setArg(1,*deviceObj->sum_d_B);
    softmax_end_kernel.setArg(2,n);

    // Enqueue both kernels
    deviceObj->queue->enqueueNDRangeKernel(softmax_kernel,cl::NullRange,global,local, NULL, deviceObj->evt);
    deviceObj->queue->enqueueNDRangeKernel(softmax_end_kernel,cl::NullRange,global,local, NULL, deviceObj->evt_complemet);
    
    // Wait for completion before stopping the clock
    deviceObj->queue->finish();
    // Clock profilling end 
    kernelCLK.end();

    // store the kernel time
    deviceObj->elapsed_time = kernelCLK.getElapsedNS();
}

