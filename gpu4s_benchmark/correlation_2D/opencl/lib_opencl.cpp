// OpenCL lib code 
#include <cmath>
#include "../benchmark_library.h"
#include <cstring>
#include "GEN_kernel.hcl"
#include "GEN_atomic_functions.hcl"


void execute_kernel(GraficCommon* device_object, unsigned int n){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    const unsigned int x_local= BLOCK_SIZE;
    const unsigned int y_local= BLOCK_SIZE;
    cl::NDRange local;
    cl::NDRange global;
    if (n <= BLOCK_SIZE )
    {
        local = cl::NullRange;
        global = cl::NDRange(n,n);
    }
    else
    {
        local = cl::NDRange(x_local,y_local);
        global = cl::NDRange(n,n);
    }
    
    cl::Program::Sources sources;
    // load kernel from file
    kernel_code = type_kernel + atomic_code + kernel_code;
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

    cl::Kernel kernel_mean=cl::Kernel(program,"mean_matrices");
    kernel_mean.setArg(0,*deviceObj->d_A);
    kernel_mean.setArg(1,*deviceObj->d_B);
    kernel_mean.setArg(2,*deviceObj->mean_A);
    kernel_mean.setArg(3,*deviceObj->mean_B);
    kernel_mean.setArg(4,n);
    deviceObj->queue->enqueueNDRangeKernel(kernel_mean,cl::NullRange,global,local, NULL, deviceObj->evt_mean);

    cl::Kernel kernel=cl::Kernel(program,"correlation_2D");
    kernel.setArg(0,*deviceObj->d_A);
    kernel.setArg(1,*deviceObj->d_B);
    kernel.setArg(2,*deviceObj->d_R);
    kernel.setArg(3,*deviceObj->mean_A);
    kernel.setArg(4,*deviceObj->mean_B);
    kernel.setArg(5,*deviceObj->acumulate_value_a_b);
    kernel.setArg(6,*deviceObj->acumulate_value_a_a);
    kernel.setArg(7,*deviceObj->acumulate_value_b_b);
    kernel.setArg(8,n);
    
    deviceObj->queue->enqueueNDRangeKernel(kernel,cl::NullRange,global,local, NULL, deviceObj->evt);
    
    // Wait for completion before stopping the clock
    deviceObj->queue->finish();
    // Clock profilling end 
    kernelCLK.end();

    // store the kernel time
    deviceObj->elapsed_time = kernelCLK.getElapsedNS();
}

