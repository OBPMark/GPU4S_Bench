// OpenCL lib code 
#include <cmath>
#include "../benchmark_library.h"
#include "GEN_kernel_opt.hcl"
#include "GEN_atomic_functions.hcl"

#define BLOCK_SIZE_PLANE BLOCK_SIZE*BLOCK_SIZE

void execute_kernel(GraficCommon* device_object, unsigned int input_data, unsigned int output_data, unsigned int kernel_1, unsigned int kernel_2, unsigned int stride_1, unsigned int stride_2, unsigned int neurons_dense_1, unsigned int neurons_dense_2){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    unsigned int x_local= BLOCK_SIZE;
    unsigned int y_local= BLOCK_SIZE;
    unsigned int x_local_plane= BLOCK_SIZE_PLANE;
    cl::NDRange local;
    cl::NDRange global;

    cl::Program::Sources sources;
    // load kernel from file
    char str[12];
    sprintf(str, "%d", BLOCK_SIZE);
    kernel_code = type_kernel_common+ "#define BLOCK_SIZE " + str + "\n" +atomic_code + kernel_code;
    sources.push_back({kernel_code.c_str(),kernel_code.length()});
    // build
    cl::Program program(*deviceObj->context,sources);
    if(program.build({deviceObj->default_device})!=CL_SUCCESS){
        std::cout<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(deviceObj->default_device)<<"\n";
        exit(1);
    }
    // 1-1 step convolution
    if (input_data <= BLOCK_SIZE)
    {
        local = cl::NullRange;
        global = cl::NDRange(input_data, input_data);
    }
    else
    {
        local = cl::NDRange(x_local, y_local);
        global = cl::NDRange(input_data, input_data);
    }

    unsigned int kernel_rad =  kernel_1 / 2;
    unsigned int size_shared = (BLOCK_SIZE + kernel_rad *2 ) * sizeof(bench_t) * (BLOCK_SIZE + kernel_rad *2) * sizeof(bench_t);
    unsigned int size_shared_position = (BLOCK_SIZE + kernel_rad *2);

    // kernel time execution
    Clock kernelCLK;

    // Clock profilling start 
    kernelCLK.start();

    // 1-1 step convolution
    cl::Kernel kernel_conv=cl::Kernel(program,"kernel_matrix_convolution");
    kernel_conv.setArg(0,*deviceObj->input_data);
    kernel_conv.setArg(1,*deviceObj->conv_1_output);
    kernel_conv.setArg(2,*deviceObj->kernel_1);
    kernel_conv.setArg(3,input_data);
    kernel_conv.setArg(4,input_data);
    kernel_conv.setArg(5,input_data);
    kernel_conv.setArg(6,kernel_1);
    kernel_conv.setArg(7, cl::Local(size_shared));
    kernel_conv.setArg(8, size_shared_position);
    kernel_conv.setArg(9, kernel_rad);

    

    deviceObj->queue->enqueueNDRangeKernel(kernel_conv,cl::NullRange,global,local, NULL, deviceObj->evt1_1);

    // 1-2 step activation
    if (input_data*input_data <= BLOCK_SIZE_PLANE)
    {
        local = cl::NullRange;
        global = cl::NDRange(input_data*input_data);
    }
    else
    {
        local = cl::NDRange(x_local_plane);
        global = cl::NDRange(input_data*input_data);
    }
    
    cl::Kernel  kernel_add=cl::Kernel(program,"kernel_relu");
    kernel_add.setArg(0,*deviceObj->conv_1_output);
    kernel_add.setArg(1,*deviceObj->conv_1_output);
    kernel_add.setArg(2,input_data);
    deviceObj->queue->enqueueNDRangeKernel(kernel_add,cl::NullRange,global,local, NULL, deviceObj->evt1_2);

    // 1-3 step pooling
    unsigned int size_lateral_1 = input_data / stride_1;
    if(size_lateral_1 <= BLOCK_SIZE)
    {
        local = cl::NullRange;
        global = cl::NDRange (size_lateral_1 *size_lateral_1);
    }
    else
    {
        local = cl::NDRange(x_local_plane);
        global = cl::NDRange(size_lateral_1 * size_lateral_1);
    }
    kernel_add=cl::Kernel(program,"kernel_max");
    kernel_add.setArg(0,*deviceObj->conv_1_output);
    kernel_add.setArg(1,*deviceObj->pooling_1_output);
    kernel_add.setArg(2,input_data);
    kernel_add.setArg(3,stride_1);
    kernel_add.setArg(4,size_lateral_1);
    deviceObj->queue->enqueueNDRangeKernel(kernel_add,cl::NullRange,global,local, NULL, deviceObj->evt1_3);
    
    // 1-4 step normalitation
    if(size_lateral_1 <= BLOCK_SIZE)
    {
        local = cl::NullRange;
        global = cl::NDRange (size_lateral_1, size_lateral_1);
    }
    else
    {
        local = cl::NDRange(x_local, y_local);
        global = cl::NDRange(size_lateral_1, size_lateral_1);
    }
    kernel_add=cl::Kernel(program,"kernel_lrn");
    kernel_add.setArg(0,*deviceObj->pooling_1_output);
    kernel_add.setArg(1,*deviceObj->pooling_1_output);
    kernel_add.setArg(2,size_lateral_1);
    kernel_add.setArg(3,K);
    kernel_add.setArg(4,ALPHA);
    kernel_add.setArg(5,BETA);
    deviceObj->queue->enqueueNDRangeKernel(kernel_add,cl::NullRange,global,local, NULL, deviceObj->evt1_4);

    // 2-1 step convolutions
    int kernel_rad_2 =  kernel_2 / 2;
    int size_shared_2 = (BLOCK_SIZE + kernel_rad_2 *2 ) * sizeof(bench_t) * (BLOCK_SIZE + kernel_rad_2 *2) * sizeof(bench_t);
    int size_shared_position_2 = (BLOCK_SIZE + kernel_rad_2 *2);    

    kernel_conv=cl::Kernel(program,"kernel_matrix_convolution_old");
    kernel_conv.setArg(0,*deviceObj->pooling_1_output);
    kernel_conv.setArg(1,*deviceObj->conv_2_output);
    kernel_conv.setArg(2,*deviceObj->kernel_2);
    kernel_conv.setArg(3,size_lateral_1);
    kernel_conv.setArg(4,size_lateral_1);
    kernel_conv.setArg(5,size_lateral_1);
    kernel_conv.setArg(6,kernel_2);
    //kernel_conv.setArg(7, cl::Local(size_shared_2));
    //kernel_conv.setArg(8, size_shared_position_2);
    //kernel_conv.setArg(9, kernel_rad_2);
    deviceObj->queue->enqueueNDRangeKernel(kernel_conv,cl::NullRange,global,local, NULL, deviceObj->evt2_1);

    // 2-2 step activation
    if (size_lateral_1*size_lateral_1 <= BLOCK_SIZE_PLANE)
    {
        local = cl::NullRange;
        global = cl::NDRange(size_lateral_1*size_lateral_1);
    }
    else
    {
        local = cl::NDRange(x_local_plane);
        global = cl::NDRange(size_lateral_1*size_lateral_1);
    }
    
    kernel_add=cl::Kernel(program,"kernel_relu");
    kernel_add.setArg(0,*deviceObj->conv_2_output);
    kernel_add.setArg(1,*deviceObj->conv_2_output);
    kernel_add.setArg(2,size_lateral_1);
    deviceObj->queue->enqueueNDRangeKernel(kernel_add,cl::NullRange,global,local, NULL, deviceObj->evt2_2);
   
    // 2-3 normalization
    if (size_lateral_1 <= BLOCK_SIZE)
    {
        local = cl::NullRange;
        global = cl::NDRange(size_lateral_1, size_lateral_1);
    }
    else
    {
        local = cl::NDRange(x_local, y_local);
        global = cl::NDRange(size_lateral_1, size_lateral_1);
    }
    
    kernel_add=cl::Kernel(program,"kernel_lrn");
    kernel_add.setArg(0,*deviceObj->conv_2_output);
    kernel_add.setArg(1,*deviceObj->conv_2_output);
    kernel_add.setArg(2,size_lateral_1);
    kernel_add.setArg(3,K);
    kernel_add.setArg(4,ALPHA);
    kernel_add.setArg(5,BETA);

    deviceObj->queue->enqueueNDRangeKernel(kernel_add,cl::NullRange,global,local, NULL, deviceObj->evt2_3);

    // 2-4 step pooling
    unsigned int size_lateral_2 = size_lateral_1 / stride_2;
    if(size_lateral_2 <= BLOCK_SIZE)
    {
        local = cl::NullRange;
        global = cl::NDRange (size_lateral_2 *size_lateral_2);
    }
    else
    {
        local = cl::NDRange(x_local_plane);
        global = cl::NDRange(size_lateral_2 * size_lateral_2);
    }
    kernel_add=cl::Kernel(program,"kernel_max");
    kernel_add.setArg(0,*deviceObj->conv_2_output);
    kernel_add.setArg(1,*deviceObj->pooling_2_output);
    kernel_add.setArg(2,size_lateral_1);
    kernel_add.setArg(3,stride_2);
    kernel_add.setArg(4,size_lateral_2);
    deviceObj->queue->enqueueNDRangeKernel(kernel_add,cl::NullRange,global,local, NULL, deviceObj->evt2_4);
    
    // dense layer 1
    if(neurons_dense_1 <= BLOCK_SIZE)
    {
        local = cl::NullRange;
        global = cl::NDRange (neurons_dense_1, 1);
    }
    else
    {
        local = cl::NullRange;
        global = cl::NDRange(neurons_dense_1, 1);
    }
    kernel_add=cl::Kernel(program,"kernel_matrix_multiplication");
    kernel_add.setArg(0,*deviceObj->dense_layer_1_weights);
    kernel_add.setArg(1,*deviceObj->pooling_2_output);
    kernel_add.setArg(2,*deviceObj->dense_layer_1_output);
    kernel_add.setArg(3,neurons_dense_1);
    kernel_add.setArg(4,1);
    kernel_add.setArg(5,size_lateral_2*size_lateral_2);

    deviceObj->queue->enqueueNDRangeKernel(kernel_add,cl::NullRange,global,local, NULL, deviceObj->evtd_1);

    //activation layer dense 1
    if ((neurons_dense_1) <= BLOCK_SIZE_PLANE)
    {
        local = cl::NullRange;
        global = cl::NDRange(neurons_dense_1);
    }
    else
    {
        local = cl::NullRange;
        global = cl::NDRange(neurons_dense_1);
    }
    kernel_add=cl::Kernel(program,"kernel_relu_linear");
    kernel_add.setArg(0,*deviceObj->dense_layer_1_output);
    kernel_add.setArg(1,*deviceObj->dense_layer_1_output);
    kernel_add.setArg(2,neurons_dense_1);
    deviceObj->queue->enqueueNDRangeKernel(kernel_add,cl::NullRange,global,local, NULL, deviceObj->evtd_1_a);

    // dense layer 2
    if(neurons_dense_2 <= BLOCK_SIZE)
    {
        local = cl::NullRange;
        global = cl::NDRange (neurons_dense_2, 1);
    }
    else
    {
        local = cl::NullRange;
        global = cl::NDRange(neurons_dense_2, 1);
    }
    kernel_add=cl::Kernel(program,"kernel_matrix_multiplication");
    kernel_add.setArg(0,*deviceObj->dense_layer_2_weights);
    kernel_add.setArg(1,*deviceObj->dense_layer_1_output);
    kernel_add.setArg(2,*deviceObj->dense_layer_2_output);
    kernel_add.setArg(3,neurons_dense_2);
    kernel_add.setArg(4,1);
    kernel_add.setArg(5,neurons_dense_1);

    deviceObj->queue->enqueueNDRangeKernel(kernel_add,cl::NullRange,global,local, NULL, deviceObj->evtd_2);

    //activation layer dense 2
    if ((neurons_dense_2) <= BLOCK_SIZE_PLANE)
    {
        local = cl::NullRange;
        global = cl::NDRange(neurons_dense_2);
    }
    else
    {
        local = cl::NDRange(x_local_plane);
        global = cl::NDRange(neurons_dense_2);
    }
    kernel_add=cl::Kernel(program,"kernel_relu_linear");
    kernel_add.setArg(0,*deviceObj->dense_layer_2_output);
    kernel_add.setArg(1,*deviceObj->dense_layer_2_output);
    kernel_add.setArg(2,neurons_dense_2);
    deviceObj->queue->enqueueNDRangeKernel(kernel_add,cl::NullRange,global,local, NULL, deviceObj->evtd_2_a);

    //soft max - output
    if((neurons_dense_2) <= BLOCK_SIZE)
    {
        local = cl::NullRange;
        global = cl::NDRange (neurons_dense_2);
    }
    else
    {
        local = cl::NDRange(x_local);
        global = cl::NDRange(neurons_dense_2);
    }
    cl::Kernel softmax_kernel=cl::Kernel(program,"kernel_softmax");
    softmax_kernel.setArg(0,*deviceObj->dense_layer_2_output);
    softmax_kernel.setArg(1,*deviceObj->output_data);
    softmax_kernel.setArg(2,*deviceObj->sum_ouput);
    softmax_kernel.setArg(3,neurons_dense_2);

    deviceObj->queue->enqueueNDRangeKernel(softmax_kernel,cl::NullRange,global,local, NULL, deviceObj->evt_softmax);

    cl::Kernel softmax_end_kernel=cl::Kernel(program,"kernel_softmax_end");
    softmax_end_kernel.setArg(0,*deviceObj->output_data);
    softmax_end_kernel.setArg(1,*deviceObj->sum_ouput);
    softmax_end_kernel.setArg(2,neurons_dense_2);

    deviceObj->queue->enqueueNDRangeKernel(softmax_end_kernel,cl::NullRange,global,local, NULL, deviceObj->evt_softmax_fin);

    // end 
    // Wait for completion before stopping the clock
    deviceObj->queue->finish();
    // Clock profilling end 
    kernelCLK.end();

    // store the kernel time
    deviceObj->elapsed_time = kernelCLK.getElapsedNS();
}
