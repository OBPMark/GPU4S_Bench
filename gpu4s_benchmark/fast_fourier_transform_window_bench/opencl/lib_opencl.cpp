// OpenCL lib code 
#include <cmath>
#include "../benchmark_library.h"
#include "GEN_kernel.hcl"

void aux_execute_kernel(GraficCommon* device_object, int64_t size, int64_t position, cl::Program program){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    size = size / 2;
    const unsigned int x_local= BLOCK_SIZE;
    unsigned int mode = (unsigned int)log2(size);
    cl::NDRange local_reverse, global_reverse, local, global;
    if (size > BLOCK_SIZE)
    {
        local_reverse =  cl::NDRange (x_local);
        global_reverse = cl::NDRange (size);
    }
    else
    {
        local_reverse = cl::NullRange;
        global_reverse = cl::NDRange(size);
    }
   

    //cl::NDRange local(x_local, y_local);
    //cl::NDRange global(n, w);
    

    // reverse bit operation 
    cl::Kernel kernel_add=cl::Kernel(program,"binary_reverse_kernel");
    kernel_add.setArg(0,*deviceObj->d_A);
    kernel_add.setArg(1,*deviceObj->d_B);
    kernel_add.setArg(2,size);
    kernel_add.setArg(3,mode);
    kernel_add.setArg(4,position);

    
    deviceObj->queue->enqueueNDRangeKernel(kernel_add,cl::NullRange,global_reverse,local_reverse, NULL, NULL);
    deviceObj->queue->finish();

    // FFT calculation
    bench_t wtemp, wr, wpr, wpi, wi, theta;
    unsigned int theads = size/2;
    unsigned int loop = 1;
    cl::Kernel kernel_fft=cl::Kernel(program,"fft_kernel");
    

    while(loop < size){
        // caluclate values 
        theta = -(M_PI/loop); // check
        wtemp = sin(0.5*theta);
        wpr = -2.0*wtemp*wtemp;
        wpi = sin(theta);
        wr = 1.0;
        wi = 0.0;
        // calculate block size and thead size
        if (theads % BLOCK_SIZE != 0){
            // inferior part
            global = cl::NDRange (theads);
            local = cl::NullRange;
        }
        else{
            // top part
            local =  cl::NDRange (x_local);
            global = cl::NDRange (theads);
        }
        // launch kernel loop times
        for(unsigned int i = 0; i < loop; ++i){
            //kernel launch 
            kernel_fft.setArg(0,*deviceObj->d_B);
            kernel_fft.setArg(1,loop);
            kernel_fft.setArg(2,i);
            kernel_fft.setArg(3,wr);
            kernel_fft.setArg(4,wi);
            kernel_fft.setArg(5,size);
            kernel_fft.setArg(6,position);
            deviceObj->queue->enqueueNDRangeKernel(kernel_fft,cl::NullRange,global,local, NULL, NULL);
            // update WR, WI
            wtemp=wr;
            wr += wr*wpr - wi*wpi;
            wi += wi*wpr + wtemp*wpi;
            
        }
        // update loop values
        loop = loop * 2;
        theads = theads / 2;
       
    }

    deviceObj->queue->finish();
}

void execute_kernel(GraficCommon* device_object, int64_t window, int64_t size){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    deviceObj->elapsed_time = 0;
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

    //FIX : GPU profiling use opencl marker
    deviceObj->queue->enqueueMarkerWithWaitList(NULL, deviceObj->evt);

    for (unsigned int i = 0; i < (size * 2 - window + 1); i+=2){
        aux_execute_kernel(device_object, window, i, program);
    }
    
    //FIX : GPU profiling use opencl marker
    deviceObj->queue->enqueueMarkerWithWaitList(NULL, deviceObj->evt_end);

    // Wait for completion before stopping the clock
    deviceObj->queue->finish();
    // Clock profilling end 
    kernelCLK.end();

    // store the kernel time
    deviceObj->elapsed_time = kernelCLK.getElapsedNS();
}
