// OpenCL lib code 
#include <cmath>
#include "../benchmark_library.h"
#include <cstring>
#include "GEN_kernel_opt.hcl"

void execute_kernel(GraficCommon* device_object, unsigned int n, unsigned int m,unsigned int w, unsigned int kernel_size){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    const unsigned int x_local= BLOCK_SIZE;
    const unsigned int y_local= BLOCK_SIZE;
    cl::NDRange local;
    cl::NDRange global;
    if (n < BLOCK_SIZE)
    {
        local = cl::NullRange;
        global = cl::NDRange(n, w);
    }
    else
    {
        local = cl::NDRange(x_local, y_local);
        global = cl::NDRange(n, w);
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

    unsigned int kernel_rad =  kernel_size / 2;
    unsigned int size_shared = (BLOCK_SIZE + kernel_rad *2 ) * sizeof(bench_t) * (BLOCK_SIZE + kernel_rad *2) * sizeof(bench_t);
    unsigned int size_shared_position = (BLOCK_SIZE + kernel_rad *2);

    // kernel time execution
    Clock kernelCLK;

    // Clock profilling start 
    kernelCLK.start();

    cl::Kernel kernel_conv=cl::Kernel(program,"kernel_matrix_convolution");
    kernel_conv.setArg(0,*deviceObj->d_A);
    kernel_conv.setArg(1,*deviceObj->d_B);
    kernel_conv.setArg(2,*deviceObj->kernel);
    kernel_conv.setArg(3,n);
    kernel_conv.setArg(4,m);
    kernel_conv.setArg(5,w);
    kernel_conv.setArg(6,kernel_size);
    kernel_conv.setArg(7, cl::Local(size_shared));
    kernel_conv.setArg(8, size_shared_position);
    kernel_conv.setArg(9, kernel_rad);

    deviceObj->queue->enqueueNDRangeKernel(kernel_conv,cl::NullRange,global,local, NULL, deviceObj->evt);
    
    // Wait for completion before stopping the clock
    deviceObj->queue->finish();
    // Clock profilling end 
    kernelCLK.end();

    // store the kernel time
    deviceObj->elapsed_time = kernelCLK.getElapsedNS();
}
